import os
import re
import io
import json
import base64
import hashlib
import urllib.parse
from pydantic import BaseModel

from fastapi import FastAPI, Form, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from dotenv import load_dotenv

import cloudinary
import cloudinary.uploader
import cloudinary.api

import lancedb
from lancedb.embeddings import get_registry
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper

from google.oauth2 import service_account
from google.cloud import storage

from typing import List, Optional, Dict, Any, Tuple
import pyarrow as pa
from lancedb.pydantic import LanceModel, Vector

load_dotenv()

# ------------------------------------------------------------------------------
# Глобальные настройки
# ------------------------------------------------------------------------------
client = OpenAI()  # берет OPENAI_API_KEY из .env
tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = 512
converter = DocumentConverter()

b64_key = os.environ["GOOGLE_CLOUD_KEY"]
decoded = base64.b64decode(b64_key)
sa_info = json.loads(decoded)
creds = service_account.Credentials.from_service_account_info(sa_info)
storage_client = storage.Client(credentials=creds, project=sa_info["project_id"])
bucket = storage_client.bucket(os.environ["GCS_BUCKET_NAME"])

# Подключаемся к LanceDB (папка "data/lancedb" должна существовать или будет создана)
db = lancedb.connect("data/lancedb")
func = get_registry().get("openai").create(name="text-embedding-3-large")

app = FastAPI()

cloudinary.config(
  cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
  api_key = os.getenv("CLOUDINARY_API_KEY"),
  api_secret = os.getenv("CLOUDINARY_API_SECRET"),
  secure=True # Recommended to use HTTPS
)

# ------------------------------------------------------------------------------
# Feature flags and config (env-driven)
# ------------------------------------------------------------------------------
def _parse_bool_env(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

RAG_USE_LLM_UNDERSTAND_DEFAULT = _parse_bool_env(os.getenv("RAG_USE_LLM_UNDERSTAND"), False)
RAG_NATURAL_CHUNKING_DEFAULT   = _parse_bool_env(os.getenv("RAG_NATURAL_CHUNKING"), True)
RAG_LLM_RESTRUCTURE_DEFAULT    = _parse_bool_env(os.getenv("RAG_LLM_RESTRUCTURE"), True)
RAG_SOFT_MAX_TOKENS            = int(os.getenv("RAG_SOFT_MAX_TOKENS", "900"))
RAG_HARD_MAX_TOKENS            = int(os.getenv("RAG_HARD_MAX_TOKENS", "1800"))
LLM_SUM_MODEL                  = os.getenv("LLM_SUM_MODEL", "gpt-4.1-mini")

# OpenAI reranker configuration
OPENAI_RERANKER_ENABLED        = _parse_bool_env(os.getenv("OPENAI_RERANKER_ENABLED"), False)
OPENAI_RERANKER_MODEL          = os.getenv("OPENAI_RERANKER_MODEL", "gpt-4o-mini")
OPENAI_RERANKER_TOP_K          = int(os.getenv("OPENAI_RERANKER_TOP_K", "20"))
OPENAI_RERANKER_TIMEOUT        = int(os.getenv("OPENAI_RERANKER_TIMEOUT", "15"))

# ------------------------------------------------------------------------------
# Sentence splitting utilities (blingfire if available, regex fallback)
# ------------------------------------------------------------------------------
try:
    import blingfire  # type: ignore
    _BLINGFIRE_AVAILABLE = True
except Exception:
    _BLINGFIRE_AVAILABLE = False

def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences without losing punctuation. Prefer blingfire if available."""
    if not text:
        return []
    try:
        if _BLINGFIRE_AVAILABLE:
            # blingfire.text_to_sentences returns sentences separated by newlines
            sents = blingfire.text_to_sentences(text).splitlines()
            # Strip but keep punctuation at end
            return [s.strip() for s in sents if s.strip()]
    except Exception:
        pass
    # Fallback: simple regex that tries to keep common sentence endings
    # This is not perfect but avoids splitting inside numbers/abbreviations most of the time
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9(])", text.strip())
    return [p.strip() for p in parts if p.strip()]

def _count_tokens(text: str) -> int:
    # Using the existing OpenAI tokenizer wrapper to approximate token count
    try:
        return len(tokenizer.tokenize(text))
    except Exception:
        # Very safe fallback
        return max(1, len(text) // 4)

# ------------------------------------------------------------------------------
# SSE helpers
# ------------------------------------------------------------------------------
def _sse_format(event: str, data: Any) -> str:
    try:
        payload = json.dumps(data, ensure_ascii=False)
    except Exception:
        payload = json.dumps({"message": str(data)})
    return f"event: {event}\ndata: {payload}\n\n"

# ------------------------------------------------------------------------------
# OpenAI reranker service
# ------------------------------------------------------------------------------
def _openai_rerank(query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Rerank documents using OpenAI GPT-4o-mini for relevance scoring.
    Returns reranked documents with updated scores.
    """
    if not OPENAI_RERANKER_ENABLED or len(documents) <= 1:
        return documents[:top_k]
    
    try:
        # Prepare documents for OpenAI with numbering
        doc_list = []
        for i, doc in enumerate(documents):
            text_preview = doc.get("text", "")[:300]  # Limit to 300 chars
            title = doc.get("title", "Без названия")
            filename = doc.get("filename", "")
            
            doc_summary = f"[{i}] Раздел: {title}\nФайл: {filename}\nТекст: {text_preview}..."
            doc_list.append(doc_summary)
        
        # Create reranking prompt
        docs_text = "\n\n".join(doc_list)
        
        prompt = f"""Вы - эксперт по поиску в медицинской базе знаний. Оцените релевантность каждого документа к запросу пользователя.

ЗАПРОС: "{query}"

ДОКУМЕНТЫ:
{docs_text}

Верните JSON со списком индексов документов, отсортированных по релевантности (от самого релевантного к менее релевантному). Включите только {min(top_k, len(documents))} самых релевантных документов.

Формат ответа:
{{"rankings": [
    {{"index": 0, "relevance_score": 0.95, "reason": "краткое объяснение"}},
    {{"index": 2, "relevance_score": 0.87, "reason": "краткое объяснение"}}
]}}"""

        response = client.chat.completions.create(
            model=OPENAI_RERANKER_MODEL,
            messages=[
                {"role": "system", "content": "Вы эксперт по медицинской информации. Оцените релевантность документов точно и объективно."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=OPENAI_RERANKER_TIMEOUT
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        reranked_docs = []
        for ranking in result.get("rankings", []):
            doc_idx = ranking.get("index")
            relevance_score = ranking.get("relevance_score", 0.0)
            reason = ranking.get("reason", "")
            
            if 0 <= doc_idx < len(documents):
                reranked_doc = documents[doc_idx].copy()
                reranked_doc["rerank_score"] = relevance_score
                reranked_doc["rerank_reason"] = reason
                # Keep original vector score for comparison
                reranked_doc["vector_score"] = reranked_doc.get("score", 0.0)
                reranked_doc["score"] = relevance_score  # Use rerank score as primary
                reranked_docs.append(reranked_doc)
        
        print(f"OpenAI reranked {len(documents)} -> {len(reranked_docs)} documents")
        return reranked_docs[:top_k]
        
    except Exception as e:
        print(f"OpenAI reranker error: {str(e)}")
        return documents[:top_k]

# ------------------------------------------------------------------------------
# Markdown-based section/paragraph extraction
# ------------------------------------------------------------------------------
def _extract_sections_from_markdown(md: str) -> List[Dict[str, Any]]:
    """
    Parse markdown into sections and paragraphs. We keep it conservative and
    language-agnostic. Returns a list of sections:
      [{
          'title': str,
          'level': int,  # heading level 1..6
          'paragraphs': [
              { 'text': str, 'para_idx': int }
          ],
          'section_idx': int
      }]
    """
    lines = md.splitlines()
    sections: List[Dict[str, Any]] = []
    current_title = None
    current_level = 1
    current_paragraph_lines: List[str] = []
    section_idx = -1
    para_idx = 0

    def _flush_paragraph():
        nonlocal para_idx, current_paragraph_lines
        if current_paragraph_lines and section_idx >= 0:
            paragraph_text = "\n".join(current_paragraph_lines).strip()
            if paragraph_text:
                sections[section_idx]["paragraphs"].append({
                    "text": paragraph_text,
                    "para_idx": para_idx,
                })
                para_idx += 1
        current_paragraph_lines = []

    def _start_new_section(title: Optional[str], level: int):
        nonlocal section_idx, para_idx
        sections.append({
            "title": title.strip() if title else None,
            "level": level,
            "paragraphs": [],
            "section_idx": len(sections),
        })
        section_idx = len(sections) - 1
        para_idx = 0

    for ln in lines:
        # Heading detection (e.g., #, ##, ###)
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            # Finish previous paragraph before starting a new section
            _flush_paragraph()
            current_title = m.group(2).strip()
            current_level = len(m.group(1))
            _start_new_section(current_title, current_level)
            continue

        # Blank line => paragraph boundary
        if not ln.strip():
            _flush_paragraph()
            continue

        # Accumulate paragraph lines; start default section if none yet
        if section_idx < 0:
            _start_new_section(title=None, level=1)
        current_paragraph_lines.append(ln)

    # Flush tail
    _flush_paragraph()

    # If no sections at all, create one default section from entire text
    if not sections and md.strip():
        sections = [{
            "title": None,
            "level": 1,
            "paragraphs": [{"text": md.strip(), "para_idx": 0}],
            "section_idx": 0,
        }]
    return sections

# ------------------------------------------------------------------------------
# Natural chunking: pack by paragraphs/sentences with soft/hard token limits
# ------------------------------------------------------------------------------
def _build_natural_chunks(
    sections: List[Dict[str, Any]],
    soft_max_tokens: int,
    hard_max_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Returns a list of chunks:
      [{ 'text': str,
         'section_title': Optional[str],
         'paragraph_range': Tuple[int, int],  # inclusive indices within section
         'sentence_range': Tuple[int, int],   # inclusive indices within packed stream
      }]
    """
    chunks: List[Dict[str, Any]] = []

    for section in sections:
        section_title = section.get("title")
        # Pack across paragraphs, but never split sentences
        current_text_parts: List[str] = []
        current_tokens = 0
        current_para_start = None
        current_sent_start = 0
        sent_counter = 0

        for para in section.get("paragraphs", []):
            para_text = para.get("text", "").strip()
            if not para_text:
                continue
            sentences = _split_into_sentences(para_text)
            if not sentences:
                continue

            # Try to add whole paragraph if possible
            para_token_count = _count_tokens(para_text)
            if para_token_count <= soft_max_tokens and (current_tokens + para_token_count) <= hard_max_tokens:
                if current_para_start is None:
                    current_para_start = para["para_idx"]
                current_text_parts.append(para_text)
                current_tokens += para_token_count
                sent_counter += len(sentences)
                continue

            # Otherwise, add sentence-by-sentence
            for si, sentence in enumerate(sentences):
                sent_tokens = _count_tokens(sentence)
                # If adding this sentence bursts the hard limit, flush current chunk first
                if current_text_parts and (current_tokens + sent_tokens) > hard_max_tokens:
                    chunks.append({
                        "text": "\n\n".join(current_text_parts),
                        "section_title": section_title,
                        "paragraph_range": (
                            current_para_start if current_para_start is not None else para["para_idx"],
                            para["para_idx"] if si == 0 else para["para_idx"],
                        ),
                        "sentence_range": (current_sent_start, sent_counter - 1),
                    })
                    current_text_parts = []
                    current_tokens = 0
                    current_para_start = None
                    current_sent_start = sent_counter

                if current_para_start is None:
                    current_para_start = para["para_idx"]
                current_text_parts.append(sentence)
                current_tokens += sent_tokens
                sent_counter += 1

                # If we crossed soft cap, consider flushing to keep chunks coherent
                if current_tokens >= soft_max_tokens:
                    chunks.append({
                        "text": "\n\n".join(current_text_parts),
                        "section_title": section_title,
                        "paragraph_range": (
                            current_para_start if current_para_start is not None else para["para_idx"],
                            para["para_idx"],
                        ),
                        "sentence_range": (current_sent_start, sent_counter - 1),
                    })
                    current_text_parts = []
                    current_tokens = 0
                    current_para_start = None
                    current_sent_start = sent_counter

        # Flush remainder for the section
        if current_text_parts:
            chunks.append({
                "text": "\n\n".join(current_text_parts),
                "section_title": section_title,
                "paragraph_range": (
                    current_para_start if current_para_start is not None else 0,
                    section["paragraphs"][len(section["paragraphs"]) - 1]["para_idx"] if section.get("paragraphs") else 0,
                ),
                "sentence_range": (current_sent_start, sent_counter - 1 if sent_counter > 0 else 0),
            })

    return chunks

# ------------------------------------------------------------------------------
# LLM document restructuring for clean knowledge base
# ------------------------------------------------------------------------------
def _llm_restructure_document(markdown_text: str) -> str:
    """
    Use LLM to restructure the entire document into clean, well-organized sections
    with clear topic separation for better chunking.
    """
    try:
        resp = client.chat.completions.create(
            model=LLM_SUM_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a knowledge base organizer. The headers you organize will be as separate chunks for an LLM's context. Try to preserve all needed context to LLM about one thing in one header.Restructure the given document into "
                        "clear, well-separated sections with proper headings. Each section should focus on "
                        "ONE specific topic. CRITICALLY IMPORTANT: Preserve ALL elements. DO NOT DELETE ANYTHING."
                        "Use markdown format with ## headings. Preserve ALL original information including "
                        "conversational scripts, pricing discussions, and sales elements. Just organize it logically by topic."
                    )
                },
                {
                    "role": "user", 
                    "content": (
                        "Restructure this document into clean sections with proper topic separation. "
                        "PRESERVE ALL conversational scripts, customer questions, staff responses, pricing info, "
                        "and sales techniques. Just organize by topic with clear ## headings:\n\n" + 
                        markdown_text
                    )
                },
            ],
            temperature=0.1,
            max_tokens=4000,
        )
        restructured = resp.choices[0].message.content if resp.choices else None
        if restructured:
            print(f"LLM restructured document: {len(restructured)} chars")
            return restructured
        else:
            print("LLM restructuring failed, using original")
            return markdown_text
    except Exception as e:
        print(f"LLM restructuring error: {e}")
        return markdown_text

# ------------------------------------------------------------------------------
# Optional: LLM understanding pass per section (best-effort, non-blocking)
# ------------------------------------------------------------------------------
def _llm_understand_sections(sections: List[Dict[str, Any]], max_sections: int = 50) -> Dict[str, Any]:
    insights: Dict[str, Any] = {"sections": []}
    used = 0
    for s in sections:
        if used >= max_sections:
            break
        section_text = "\n\n".join(p.get("text", "") for p in s.get("paragraphs", []))
        if not section_text.strip():
            continue
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a precise document analyst. Do not invent information."},
                    {"role": "user", "content": (
                        "Summarize the following section conservatively in JSON with keys: summary (1-2 sentences), "
                        "key_terms (up to 10 terms), acronyms (map of acronym->expansion).\n\n" + section_text
                    )},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content if resp.choices else None
            data = json.loads(content) if content else {}
            insights["sections"].append({
                "section_idx": s.get("section_idx"),
                "title": s.get("title"),
                "insight": data,
            })
            used += 1
        except Exception:
            # Non-blocking: ignore failures
            continue
    return insights

# ------------------------------------------------------------------------------
# Pydantic-схемы для таблиц
# ------------------------------------------------------------------------------
class ChunkMetadata(LanceModel):
    filename: Optional[str] = None
    page_numbers: Optional[str] = None  # Store as JSON string instead of List[int]
    title: Optional[str] = None
    url: str

class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadata

class UpdateSendableDescription(BaseModel):
    new_description: str

# Alternative approach - use a separate field for page numbers
class ChunkMetadataAlt(LanceModel):
    filename: Optional[str] = None
    title: Optional[str] = None
    url: str

class ChunksAlt(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadataAlt
    page_numbers: Optional[str] = None 

class SendableMetadata(LanceModel):
    index: Optional[str] = None 
    file_type: str | None
    filename: str | None
    url: str | None # GCS URL
    cloudinary_url: Optional[str] = None
    cloudinary_public_id: Optional[str] = None

class SendableFile(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: Optional[SendableMetadata] = None

# ------------------------------------------------------------------------------
# Хелпер-функции
# ------------------------------------------------------------------------------
def get_company_doc_table(company_id: str):
    """Returns or creates a table for the company's documents."""
    table_name = f"docling_{company_id}"
    try:
        table = db.open_table(table_name)
        print(f"Opened existing table '{table_name}'.")
    except ValueError:
        table = db.create_table(table_name, schema=Chunks, mode="create")
        print(f"Created new table '{table_name}'.")
    return table

def get_company_sendable_table(company_id: str):
    """Returns or creates a table for the company's sendable files."""
    table_name = f"sendable_files_{company_id}"
    try:
        table = db.open_table(table_name)
        print(f"Opened existing table '{table_name}'.")
    except ValueError:
        table = db.create_table(table_name, schema=SendableFile, mode="create")
        print(f"Created new table '{table_name}'.")
    return table

def safe_decode_filename(filename: str) -> str:
    """Fixes improperly decoded filenames."""
    try:
        # First try URL decoding in case the filename is URL-encoded
        decoded = urllib.parse.unquote(filename)
        # Then try to encode/decode to handle any remaining encoding issues
        return decoded.encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        # If any encoding/decoding fails, return the URL-decoded version
        return urllib.parse.unquote(filename)

def upload_to_gcs(content: bytes, company_id: str, folder: str, filename: str, content_type: str) -> str:
    """Uploads content to Google Cloud Storage."""
    path = f"uploads/{company_id}/{folder}/{filename}"
    blob = bucket.blob(path)
    blob.upload_from_string(content, content_type=content_type)
    return f"gs://{bucket.name}/{path}"

@app.delete("/{companyId}/fuckdrop") # для удаления всех таблиц компании
def clear_company_tables(companyId: str):
    """Clear existing tables for a company to fix schema issues."""
    doc_table_name = f"docling_{companyId}"
    sendable_table_name = f"sendable_files_{companyId}"
    
    try:
        # Delete document table if it exists
        try:
            db.drop_table(doc_table_name)
            print(f"Dropped table '{doc_table_name}'")
        except ValueError:
            print(f"Table '{doc_table_name}' doesn't exist")
            
        # Delete sendable files table if it exists
        try:
            db.drop_table(sendable_table_name)
            print(f"Dropped table '{sendable_table_name}'")
        except ValueError:
            print(f"Table '{sendable_table_name}' doesn't exist")
            
    except Exception as e:
        print(f"Error clearing tables: {str(e)}")

# ------------------------------------------------------------------------------
# Document Processing Endpoint
# ------------------------------------------------------------------------------
@app.post("/companies/{companyId}/process-document")
async def process_document(
    companyId: str,
    file: UploadFile = File(...),
    llmEnrich: Optional[bool] = Query(None, description="Use LLM understanding pass"),
    naturalChunking: Optional[bool] = Query(None, description="Use paragraph/sentence natural chunking"),
    llmRestructure: Optional[bool] = Query(None, description="Use LLM to restructure document for clean topic separation"),
    softMaxTokens: Optional[int] = Query(None, description="Soft token cap for natural chunks"),
    hardMaxTokens: Optional[int] = Query(None, description="Hard token cap for natural chunks"),
    stream: Optional[bool] = Query(False, description="Stream processing updates via SSE"),
):
    safe_filename = safe_decode_filename(file.filename)
    
    # 1) Check for duplicates
    table = get_company_doc_table(companyId)
    df = table.to_pandas()
    dup_count = int(
        df["metadata"]
          .apply(lambda md: md.get("filename") == safe_filename)
          .sum()
    )
    if dup_count > 0:
        raise HTTPException(
            status_code=400,
            detail=f"File '{safe_filename}' was already processed for company '{companyId}'."
        )

    try:
        # Read file content once at the beginning
        content = await file.read()
        
        # Prepare a generator for SSE if streaming is requested
        async def _stream_generator():
            try:
                # 2) Save file locally first
                yield _sse_format("status", {"step": "save_local", "message": "Saving uploaded file"})
                upload_dir = os.path.join("uploads", companyId, "temp")
                os.makedirs(upload_dir, exist_ok=True)
                temp_path = os.path.join(upload_dir, safe_filename)
                with open(temp_path, "wb") as f:
                    f.write(content)

                # 3) Upload to GCS
                yield _sse_format("status", {"step": "upload_gcs", "message": "Uploading to GCS"})
                gcs_url = upload_to_gcs(
                    content=content,
                    company_id=companyId,
                    folder="documents",
                    filename=safe_filename,
                    content_type=file.content_type or "application/octet-stream"
                )

                # 4) Process with Docling
                yield _sse_format("status", {"step": "docling_convert", "message": "Converting with Docling"})
                result = converter.convert(source=temp_path)
                document = result.document
                markdown_output = document.export_to_markdown()
                yield _sse_format("debug", {"markdown_preview": markdown_output[:500]})

                # Resolve flags
                use_llm = llmEnrich if llmEnrich is not None else RAG_USE_LLM_UNDERSTAND_DEFAULT
                use_natural = naturalChunking if naturalChunking is not None else RAG_NATURAL_CHUNKING_DEFAULT
                use_restructure = llmRestructure if llmRestructure is not None else RAG_LLM_RESTRUCTURE_DEFAULT
                soft_cap = int(softMaxTokens) if softMaxTokens else RAG_SOFT_MAX_TOKENS
                hard_cap = int(hardMaxTokens) if hardMaxTokens else RAG_HARD_MAX_TOKENS

                # Optional LLM document restructuring for clean topic separation
                final_markdown = markdown_output
                if use_restructure:
                    yield _sse_format("status", {"step": "llm_restructure", "message": "Restructuring document with LLM for clean topic separation"})
                    final_markdown = _llm_restructure_document(markdown_output)
                    yield _sse_format("debug", {"restructured_preview": final_markdown[:500]})

                # Extract sections
                yield _sse_format("status", {"step": "extract_sections", "message": "Extracting sections from markdown"})
                sections = _extract_sections_from_markdown(final_markdown)
                yield _sse_format("debug", {"sections_count": len(sections)})

                # Optional LLM insights
                insights = None
                if use_llm:
                    yield _sse_format("status", {"step": "llm_insights", "message": "Running LLM insights (best-effort)"})
                    insights = _llm_understand_sections(sections)
                    yield _sse_format("debug", {"insights_preview": insights.get("sections", [])[:2] if insights else None})

                # Chunking
                yield _sse_format("status", {"step": "chunking", "message": "Building chunks"})
                chunks = None
                if use_natural:
                    try:
                        chunks = _build_natural_chunks(sections, soft_cap, hard_cap)
                        yield _sse_format("debug", {"natural_chunks": len(chunks)})
                    except Exception as nerr:
                        yield _sse_format("warn", {"message": f"Natural chunking failed, fallback: {str(nerr)}"})
                if chunks is None:
                    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)
                    dl_chunks = list(chunker.chunk(dl_doc=document))
                    chunks = dl_chunks
                    yield _sse_format("debug", {"hybrid_chunks": len(dl_chunks)})

                if not chunks:
                    raise ValueError("No chunks were created from the document")

                # Prepare chunks for storage and log each
                yield _sse_format("status", {"step": "prepare", "message": "Preparing chunks for storage"})
                to_store = []
                if use_natural and isinstance(chunks, list) and chunks and isinstance(chunks[0], dict):
                    for i, ch in enumerate(chunks):
                        item = {
                            "text": ch.get("text", ""),
                            "metadata": {
                                "filename": safe_filename,
                                "page_numbers": None,
                                "title": ch.get("section_title"),
                                "url": gcs_url,
                            },
                        }
                        to_store.append(item)
                        yield _sse_format("chunk", {"index": i, "title": item["metadata"]["title"], "preview": item["text"][:160]})
                else:
                    for i, chunk in enumerate(chunks):
                        page_nums = sorted({prov.page_no for item in chunk.meta.doc_items for prov in item.prov}) or []
                        page_nums_str = json.dumps(page_nums) if page_nums else None
                        item = {
                            "text": chunk.text,
                            "metadata": {
                                "filename": safe_filename,
                                "page_numbers": page_nums_str,
                                "title": chunk.meta.headings[0] if chunk.meta.headings else None,
                                "url": gcs_url,
                            },
                        }
                        to_store.append(item)
                        yield _sse_format("chunk", {"index": i, "title": item["metadata"]["title"], "preview": item["text"][:160], "pages": page_nums})

                if not to_store:
                    raise ValueError("No chunks were prepared for storage")

                # Store
                yield _sse_format("status", {"step": "store", "message": f"Storing {len(to_store)} chunks"})
                table.add(to_store)
                yield _sse_format("done", {
                    "message": "Document processed and embeddings stored successfully.",
                    "row_count": table.count_rows(),
                    "url": gcs_url,
                    "natural": bool(use_natural),
                    "llm": bool(use_llm),
                    "restructured": bool(use_restructure),
                })
            except Exception as e:
                yield _sse_format("error", {"message": str(e)})
            finally:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        if stream:
            return StreamingResponse(_stream_generator(), media_type="text/event-stream")

        # Non-streaming path (existing behavior with extra per-chunk logging)
        # 2) Save file locally first
        upload_dir = os.path.join("uploads", companyId, "temp")
        os.makedirs(upload_dir, exist_ok=True)
        temp_path = os.path.join(upload_dir, safe_filename)
        with open(temp_path, "wb") as f:
            f.write(content)

        # 3) Upload to GCS
        gcs_url = upload_to_gcs(
            content=content,
            company_id=companyId,
            folder="documents",
            filename=safe_filename,
            content_type=file.content_type or "application/octet-stream"
        )

        # 4) Process with Docling
        result = converter.convert(source=temp_path)
        document = result.document
        markdown_output = document.export_to_markdown()

        # Resolve flags
        use_llm = llmEnrich if llmEnrich is not None else RAG_USE_LLM_UNDERSTAND_DEFAULT
        use_natural = naturalChunking if naturalChunking is not None else RAG_NATURAL_CHUNKING_DEFAULT
        use_restructure = llmRestructure if llmRestructure is not None else RAG_LLM_RESTRUCTURE_DEFAULT
        soft_cap = int(softMaxTokens) if softMaxTokens else RAG_SOFT_MAX_TOKENS
        hard_cap = int(hardMaxTokens) if hardMaxTokens else RAG_HARD_MAX_TOKENS

        # Optional LLM document restructuring for clean topic separation
        final_markdown = markdown_output
        if use_restructure:
            final_markdown = _llm_restructure_document(markdown_output)

        # Extract sections
        sections = _extract_sections_from_markdown(final_markdown)

        # Optional LLM understanding
        insights = _llm_understand_sections(sections) if use_llm else None

        # Chunking
        chunks = None
        if use_natural:
            try:
                chunks = _build_natural_chunks(sections, soft_cap, hard_cap)
            except Exception as nerr:
                print(f"Natural chunking failed, fallback: {nerr}")
        if chunks is None:
            chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)
            chunks = list(chunker.chunk(dl_doc=document))

        if not chunks:
            raise ValueError("No chunks were created from the document")

        # Prepare chunks for storage with logging
        to_store = []
        if use_natural and isinstance(chunks, list) and chunks and isinstance(chunks[0], dict):
            for i, ch in enumerate(chunks):
                item = {
                    "text": ch.get("text", ""),
                    "metadata": {
                        "filename": safe_filename,
                        "page_numbers": None,
                        "title": ch.get("section_title"),
                        "url": gcs_url,
                    },
                }
                to_store.append(item)
                print({"chunk_index": i, "title": item["metadata"]["title"], "preview": item["text"][:160]})
        else:
            for i, chunk in enumerate(chunks):
                page_nums = sorted({prov.page_no for item in chunk.meta.doc_items for prov in item.prov}) or []
                page_nums_str = json.dumps(page_nums) if page_nums else None
                item = {
                    "text": chunk.text,
                    "metadata": {
                        "filename": safe_filename,
                        "page_numbers": page_nums_str,
                        "title": chunk.meta.headings[0] if chunk.meta.headings else None,
                        "url": gcs_url,
                    },
                }
                to_store.append(item)
                print({"chunk_index": i, "title": item["metadata"]["title"], "preview": item["text"][:160], "pages": page_nums})

        if not to_store:
            raise ValueError("No chunks were prepared for storage")

        table.add(to_store)
        response_obj = {
            "message": "Document processed and embeddings stored successfully.",
            "row_count": table.count_rows(),
            "url": gcs_url,
            "natural": bool(use_natural),
            "llm": bool(use_llm),
            "restructured": bool(use_restructure),
        }
        if use_llm and insights:
            response_obj["insights_preview"] = insights.get("sections", [])[:2]
        return response_obj
        
    except Exception as e:
        print(f"Document processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
    finally:
        # Clean up temp file
        try:
            if 'temp_path' in locals():
                os.remove(temp_path)
                print(f"Cleaned up temporary file: {temp_path}")
        except Exception as cleanup_error:
            print(f"Warning: Failed to clean up temporary file: {cleanup_error}")

@app.get("/companies/{companyId}/search")
async def search_documents(
    companyId: str, 
    query: str = Query(...), 
    limit: int = Query(5),
    useReranker: bool = Query(True, description="Use OpenAI GPT-4o-mini reranker for better results"),
    rerankTopK: Optional[int] = Query(None, description="Number of docs to retrieve before reranking")
):
    """
    Search through company documents using semantic search.
    Returns matching chunks with their metadata and relevance scores.
    """
    table = get_company_doc_table(companyId)

    # 1) Get embedding for the query
    emb_resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    query_vec = emb_resp.data[0].embedding

    # 2) Determine search limits
    rerank_enabled = useReranker and OPENAI_RERANKER_ENABLED
    search_limit = rerankTopK or (OPENAI_RERANKER_TOP_K if rerank_enabled else limit)
    
    # 3) Search using the vector (get more results if reranking)
    try:
        df = table.search(query_vec).limit(search_limit).to_pandas()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    # 4) Format initial results
    initial_results = []
    for _, r in df.iterrows():
        md = r["metadata"]
        # Parse page numbers from JSON string if present
        page_numbers = json.loads(md["page_numbers"]) if md["page_numbers"] else None
        
        initial_results.append({
            "text":         r["text"],
            "filename":     md["filename"],
            "page_numbers": page_numbers,
            "title":        md["title"],
            "url":          md["url"],
            "score":        float(r.get("_distance", 0))  # Vector similarity score
        })
    
    # 5) Apply Jina reranking if enabled
    final_results = initial_results
    rerank_metadata = {"reranked": False}
    
    if rerank_enabled and len(initial_results) > 1:
        import time
        start_time = time.time()
        
        try:
            reranked_results = _openai_rerank(query, initial_results, top_k=limit)
            if reranked_results:
                final_results = reranked_results
                rerank_metadata = {
                    "reranked": True,
                    "reranker_model": OPENAI_RERANKER_MODEL,
                    "original_count": len(initial_results),
                    "reranked_count": len(final_results),
                    "rerank_time_ms": int((time.time() - start_time) * 1000)
                }
        except Exception as rerank_error:
            print(f"Reranking failed, using vector results: {rerank_error}")
            final_results = initial_results[:limit]
    else:
        final_results = initial_results[:limit]
    
    return {
        "results": final_results,
        **rerank_metadata
    }

@app.delete("/companies/{companyId}/delete-document")
async def delete_document(companyId: str, filename: str = Query(...)):
    """
    Delete a document and its chunks from both LanceDB and GCS.
    The filename parameter should be just the filename, not a JSON object.
    """
    # 1) Get the table
    table = get_company_doc_table(companyId)

    # 2) Count matching records
    df = table.to_pandas()
    deleted = int(
        df["metadata"]
          .apply(lambda md: md.get("filename") == filename)
          .sum()
    )

    # 3) Delete from LanceDB
    q = f"metadata.filename = '{filename}'"
    table.delete(q)

    # 4) Delete from GCS
    path = f"uploads/{companyId}/documents/{filename}"
    try:
        bucket.blob(path).delete()
    except Exception as e:
        # Log the error but don't fail the request since the DB deletion succeeded
        print(f"Warning: Failed to delete file from GCS: {str(e)}")

    return {
        "message": f"Deleted {deleted} chunks from LanceDB and attempted to remove file from GCS.",
        "rows_deleted": deleted
    }

# ------------------------------------------------------------------------------
#  GET /companies/{companyId}/download-document
#    — вместо локальной выдачи возвращаем signed URL
# ------------------------------------------------------------------------------
@app.get("/companies/{companyId}/download-document")
async def download_document(companyId: str, filename: str = Query(...)):
    path = f"uploads/{companyId}/documents/{filename}"
    blob = bucket.blob(path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="File not found in GCS")

    signed_url = blob.generate_signed_url(expiration=15 * 60)  # 15 минут
    return JSONResponse({"url": signed_url})

# ------------------------------------------------------------------------------
# Эндпоинты для «sendable-files» (sendable_files_{companyId})
# ------------------------------------------------------------------------------

@app.post("/companies/{companyId}/process-sendable-file")
async def process_sendable_file(
    companyId:   str,
    file:        UploadFile = File(...),
    description: str        = Form(...)
):
    safe_name = safe_decode_filename(file.filename)
    table     = get_company_sendable_table(companyId)

    # 1) Check for duplicates and get existing data
    df = table.to_pandas()
    if df["metadata"].apply(lambda md: md.get("filename") == safe_name).any():
        raise HTTPException(
            status_code=400,
            detail=f"Sendable '{safe_name}' already processed for company {companyId}"
        )

    ### NEW: Generate a unique, sequential index for the sendable ###
    # 1.1) Calculate the next index
    max_index = 0
    if not df.empty:
        # Extract numeric part from indices like 'SENDABLE-123'
        # This is robust and handles cases where the index key might be missing
        # or malformed in older records.
        all_indices = []
        for md in df["metadata"]:
            index_str = md.get("index")
            if index_str and index_str.startswith("SF-"):
                try:
                    # Extract the number part and convert to integer
                    num = int(index_str.split('-')[1])
                    all_indices.append(num)
                except (ValueError, IndexError):
                    # Ignore malformed indices
                    pass
        
        if all_indices:
            max_index = max(all_indices)

    # The new index is the highest existing index + 1
    new_index_num = max_index + 1
    sendable_index = f"SF-{new_index_num}"
    ### END NEW ###


    # 2) Read content and upload to GCS
    content = await file.read()
    gcs_url = upload_to_gcs(
        content,
        companyId,
        "sendables",
        safe_name,
        file.content_type or "application/octet-stream"
    )

    # 2.1) If the file is an image, upload to Cloudinary as well
    cloudinary_url = None
    cloudinary_public_id = None
    if file.content_type and file.content_type.startswith("image/"):
        try:
            folder = f"uploads/{companyId}/sendables"
            upload_result = cloudinary.uploader.upload(
                content,
                folder=folder,
                public_id=os.path.splitext(safe_name)[0],
                resource_type="image"
            )
            cloudinary_url = upload_result.get("secure_url")
            cloudinary_public_id = upload_result.get("public_id")
        except Exception as e:
            print(f"Error uploading to Cloudinary: {e}")


    # 3) Generate embedding for the description
    emb_resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=description
    )
    embedding = emb_resp.data[0].embedding

    # 4) Prepare record and save to LanceDB
    ### MODIFIED ###
    metadata_obj = SendableMetadata(
        index=sendable_index,
        file_type=(file.content_type or os.path.splitext(safe_name)[1]),
        filename=safe_name,
        url=gcs_url
    )

    if cloudinary_url:
        metadata_obj.cloudinary_url = cloudinary_url
        metadata_obj.cloudinary_public_id = cloudinary_public_id

    # В запись передаем уже готовый, валидный объект
    record = {
        "text":    description,
        "vector":  embedding,
        "metadata": metadata_obj.model_dump() 
    }
    table.add([record])

    ### MODIFIED ###
    return {
        "message":   "Sendable file processed and saved.",
        "row_count": table.count_rows(),
        "index":     sendable_index, # Return the new index in the response
        "url": gcs_url,
        "cloudinary_url": cloudinary_url
    }

# The update endpoint does not need any changes.
# The index is a permanent identifier stored in metadata.
# This endpoint correctly only updates the 'text' and 'vector',
# leaving the original metadata (including the index) intact.
@app.put("/companies/{companyId}/update-sendable-description")
async def update_sendable_description(
    companyId: str,
    data: UpdateSendableDescription,
    filename: str = Query(...),
   
):
    """
    Updates the description for a sendable file.
    """
    table = get_company_sendable_table(companyId)
    safe_name = safe_decode_filename(filename)

    new_description = data.new_description

    # 1) Find the record to update
    df = table.to_pandas()
    rows_to_update = df[df["metadata"].apply(lambda md: md.get("filename") == safe_name)]

    if len(rows_to_update) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No sendable with filename '{safe_name}' found."
        )
    elif len(rows_to_update) > 1:
        # This should ideally never happen, but handle the edge case
        raise HTTPException(
            status_code=500,
            detail=f"Multiple sendables with filename '{safe_name}' found.  Data integrity issue."
        )

    # Get the row's LanceDB internal ID to facilitate the update operation
    record_id = rows_to_update.index[0]

    # 2) Recalculate embedding for the new description
    emb_resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=new_description
    )
    new_embedding = emb_resp.data[0].embedding

    # 3) Perform the update
    try:
        # Update the row in LanceDB. This only updates the text and vector.
        table.update(
            where=f"metadata.filename == '{safe_name}'",
            values={"text": new_description, "vector": new_embedding} # Update both text and vector
        )

        return {"message": f"Description for '{safe_name}' updated successfully."}

    except Exception as e:
        print(f"Update failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update description: {e}"
        )

@app.get("/companies/{companyId}/search-sendable")
async def search_sendable(
    companyId: str,
    query:     str   = Query(...),
    limit:     int   = Query(5)
):
    table = get_company_sendable_table(companyId)

    # 1) Получаем embedding для запроса
    emb_resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    query_vec = emb_resp.data[0].embedding

    # 2) Ищем по вектору
    df = table.search(query_vec).limit(limit).to_pandas()

    # 3) Формируем ответ
    results = []
    for _, row in df.iterrows():
        md = row["metadata"]
        results.append({
            "index":     md.get("index"),
            "text":      row["text"],
            "filename":  md["filename"],
            "file_type": md["file_type"],
            "url":       md["url"],
            "score":     float(row.get("_distance", 0))
        })

    return {"results": results}


@app.get("/companies/{companyId}/search-sendable-index")
async def search_sendable_by_index(
    companyId: str,
    query:     str = Query(..., description="The exact index to search for, e.g., 'SF-1' or 'SF-45'"),
    # The limit is kept for consistency, but for an index search, we expect only one result.
    limit:     int = Query(1) 
):
    """
    Searches for a single sendable file by its unique string index (e.g., 'SF-1').
    This is a direct metadata filter, not a vector search.
    """
    table = get_company_sendable_table(companyId)

    # 1) Build a filter query to find the exact index in the metadata.
    #    This is much faster than a vector search.
    filter_q = f"metadata.index = '{query}'"

    # 2) Execute the search using the .where() clause.
    #    .to_list() is lightweight for retrieving a small number of records.
    found_records = table.search().where(filter_q).limit(limit).to_list()

    # 3) Check if a record was found.
    if not found_records:
        raise HTTPException(
            status_code=404,
            detail=f"No sendable file found with index '{query}' for company {companyId}."
        )

    # 4) Format and return the single result.
    #    The client-side code expects the data object directly.
    record = found_records[0]
    metadata = record.get("metadata", {})
    
    # We construct a response object similar to the vector search for consistency.
    result = {
        "index":     metadata.get("index"),
        "text":      record.get("text"),
        "filename":  metadata.get("filename"),
        "file_type": metadata.get("file_type"),
        "url":       metadata.get("url"),
        "cloudinary_url": metadata.get("cloudinary_url")
    }

    return result
# ==============================================================================
#  MODIFIED: delete_sendable
# ==============================================================================
@app.delete("/companies/{companyId}/delete-sendable")
async def delete_sendable(
    companyId: str,
    filename:  Optional[str] = Query(None),
    index:     Optional[str] = Query(None)
):
    ### NEW: Validate input ###
    # Ensure exactly one identifier is provided
    if not (filename or index) or (filename and index):
        raise HTTPException(
            status_code=400,
            detail="You must provide exactly one of 'filename' or 'index'."
        )

    table = get_company_sendable_table(companyId)

    # 1) Determine filter query and user-facing identifier
    if filename:
        safe_name = safe_decode_filename(filename)
        filter_q = f"metadata.filename = '{safe_name}'"
        identifier = f"filename '{safe_name}'"
    else: # We know index is not None here because of the validation above
        filter_q = f"metadata.index = '{index}'"
        identifier = f"index '{index}'"

    # 2) Find the record(s) to delete *before* deleting from the DB
    # We need the metadata to clean up GCS and Cloudinary
    # Using .search().where() is more efficient than loading the whole table to pandas
    records_to_delete = table.search().where(filter_q).to_list()
    
    deleted_count = len(records_to_delete)
    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No sendable found with {identifier} for deletion."
        )

    # 3) Delete associated cloud assets (GCS and Cloudinary)
    for record in records_to_delete:
        metadata = record.get("metadata", {})
        
        # 3.1) Delete from Cloudinary if applicable
        public_id = metadata.get("cloudinary_public_id")
        if public_id:
            try:
                cloudinary.uploader.destroy(public_id, resource_type="image")
                print(f"Successfully deleted {public_id} from Cloudinary.")
            except Exception as e:
                print(f"Error deleting {public_id} from Cloudinary: {e}")

        # 3.2) Delete blob from GCS
        gcs_filename = metadata.get("filename")
        if gcs_filename:
            path = f"uploads/{companyId}/sendables/{gcs_filename}"
            try:
                bucket.blob(path).delete()
                print(f"Successfully deleted {path} from GCS.")
            except Exception as e:
                print(f"Failed to delete {path} from GCS: {e}")

    # 4) Delete from LanceDB (now that cloud assets are gone)
    table.delete(filter_q)

    return {
        "message": f"Successfully deleted {deleted_count} record(s) and associated files for {identifier}.",
        "rows_deleted": deleted_count
    }
# ------------------------------------------------------------------------------
# Транскрипция без сохранения
# ------------------------------------------------------------------------------
@app.post("/companies/{companyId}/transcribe-document")
async def transcribe_document(companyId: str, file: UploadFile = File(...)):
    """
    Берём файл, прогоняем через Docling, возвращаем текст. Не храним в LanceDB.
    """
    upload_dir = os.path.join("uploads", companyId, "transcriptions")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        result = converter.convert(file_path)
        document = result.document
        transcription_text = document.export_to_markdown()
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error transcribing document: {str(e)}")

    
    os.remove(file_path)
    return {"transcription": transcription_text}

# ------------------------------------------------------------------------------
# GET /companies/{companyId}/sendables/download
# ------------------------------------------------------------------------------
@app.get("/companies/{companyId}/sendables/download")
async def download_sendable(
    companyId: str,
    filename:  str = Query(...)
):
    path = f"uploads/{companyId}/sendables/{filename}"
    blob = bucket.blob(path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="Файл не найден в GCS")

    signed_url = blob.generate_signed_url(expiration=15 * 60)  # 15 минут
    return {"url": signed_url}

@app.get("/companies/{companyId}/documents/list")
async def list_documents(companyId: str):
    """
    Returns a list of documents for a company from LanceDB,
    including each file's name and metadata.
    """
    # 1) Open the LanceDB table for this company
    table = get_company_doc_table(companyId)
    
    # 2) Convert all rows to a DataFrame
    df = table.to_pandas()
    
    # 3) Group by filename to get unique documents
    documents = {}
    for _, row in df.iterrows():
        metadata = row.get('metadata') or {}
        filename = row["metadata"]["filename"]
        if filename not in documents:
            documents[filename] = {
                "filename": filename,
                "title": row["metadata"]["title"],
                "url": row["metadata"]["url"]
            }
    
    # 4) Convert to list
    results = list(documents.values())
    
    return {"files": results}

@app.get("/companies/{companyId}/sendables/list")
async def list_sendables(companyId: str):
    sendable_table = get_company_sendable_table(companyId)
    df = sendable_table.to_pandas()
    
    results = []
    for _, row in df.iterrows():
        metadata = row['metadata']
        result_item = {
            "index": metadata.get("index"),
            "filename": metadata.get("filename"),
            "text": row["text"],
            "file_type": metadata.get("file_type"),
            "url": metadata.get("url"), # GCS URL
            "cloudinary_url": metadata.get("cloudinary_url")

        }
        results.append(result_item)

    return {"sendables": results}

@app.get("/companies/{companyId}/documents/content")
async def get_all_document_content(companyId: str):
    """
    Returns the content of all documents for a company, including their text and metadata.
    This is useful for bulk processing or AI summarization of the entire document base.
    """
    table = get_company_doc_table(companyId)
    
    # Get all documents from LanceDB
    df = table.to_pandas()
    
    # Group chunks by filename to reconstruct full documents
    documents = {}
    for _, row in df.iterrows():
        filename = row["metadata"]["filename"]
        if filename not in documents:
            documents[filename] = {
                "filename": filename,
                "title": row["metadata"]["title"],
                "url": row["metadata"]["url"],
                "chunks": []
            }
        documents[filename]["chunks"].append({
            "text": row["text"],
            "page_numbers": row["metadata"]["page_numbers"]
        })
    
    # Convert to list and sort chunks by page numbers
    result = []
    for doc in documents.values():
        # Sort chunks by page numbers if available
        if doc["chunks"][0]["page_numbers"]:
            doc["chunks"].sort(key=lambda x: min(x["page_numbers"]) if x["page_numbers"] else float('inf'))
        
        # Combine all chunks into full text
        full_text = " ".join(chunk["text"] for chunk in doc["chunks"])
        result.append({
            "filename": doc["filename"],
            "title": doc["title"],
            "url": doc["url"],
            "content": full_text
        })
    print(result)
    return {"documents": result}



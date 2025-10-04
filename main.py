import os
import re
import io
import json
import base64
import hashlib
import urllib.parse
from pydantic import BaseModel

from fastapi import FastAPI, Form, UploadFile, File, Query, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from dotenv import load_dotenv

import cloudinary
import cloudinary.uploader
import cloudinary.api

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper
from utils.reranker import rerank_with_openai, rerank_with_jina

from google.oauth2 import service_account
from google.cloud import storage

from typing import List, Optional, Dict, Any, Tuple
import time
import logging

load_dotenv()

# ------------------------------------------------------------------------------
# Глобальные настройки
# ------------------------------------------------------------------------------
client = OpenAI()  # берет OPENAI_API_KEY из .env
tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = 512
converter = DocumentConverter()

# OpenAI embedding configuration
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
EMBED_DIMS = int(os.getenv("OPENAI_EMBED_DIMS", "3072"))

b64_key = os.environ["GOOGLE_CLOUD_KEY"]
decoded = base64.b64decode(b64_key)
sa_info = json.loads(decoded)
creds = service_account.Credentials.from_service_account_info(sa_info)
storage_client = storage.Client(credentials=creds, project=sa_info["project_id"])
bucket = storage_client.bucket(os.environ["GCS_BUCKET_NAME"])

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
qdrant: QdrantClient = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    timeout=60
)

app = FastAPI()

cloudinary.config(
  cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
  api_key = os.getenv("CLOUDINARY_API_KEY"),
  api_secret = os.getenv("CLOUDINARY_API_SECRET"),
  secure=True # Recommended to use HTTPS
)

# ------------------------------------------------------------------------------
# Request timing middleware
# ------------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-ms"] = f"{process_time_ms:.2f}"
    response.headers["Server-Timing"] = f"app;dur={process_time_ms:.2f}"
    logging.getLogger("uvicorn.error").info(
        f"{request.method} {request.url.path} completed in {process_time_ms:.2f} ms"
    )
    return response

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
OPENAI_RERANKER_MODEL          = os.getenv("OPENAI_RERANKER_MODEL", "gpt-4.1-nano")
OPENAI_RERANKER_TOP_K          = int(os.getenv("OPENAI_RERANKER_TOP_K", "20"))
OPENAI_RERANKER_TIMEOUT        = int(os.getenv("OPENAI_RERANKER_TIMEOUT", "15"))

# Jina reranker configuration
JINA_RERANKER_ENABLED          = _parse_bool_env(os.getenv("JINA_RERANKER_ENABLED"), False)
JINA_API_KEY                   = os.getenv("JINA_API_KEY")
JINA_RERANKER_MODEL            = os.getenv("JINA_RERANKER_MODEL", "jina-reranker-v2-base-multilingual")
JINA_RERANKER_ENDPOINT         = os.getenv("JINA_RERANKER_ENDPOINT", "https://api.jina.ai/v1/rerank")
JINA_RERANKER_TIMEOUT          = int(os.getenv("JINA_RERANKER_TIMEOUT", "15"))

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

## Reranker moved to utils/reranker.py

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
# Qdrant helper functions
# ------------------------------------------------------------------------------

def company_doc_collection(company_id: str) -> str:
    return f"docling_{company_id}"

def company_sendable_collection(company_id: str) -> str:
    return f"sendable_files_{company_id}"

def ensure_collection(name: str):
    """Create Qdrant collection if missing with HNSW index tuned for latency."""
    try:
        qdrant.get_collection(name)
        return
    except Exception:
        pass

    qdrant.recreate_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=EMBED_DIMS,
            distance=qmodels.Distance.COSINE,
        ),
        hnsw_config=qmodels.HnswConfigDiff(m=32, ef_construct=256),
        optimizers_config=qmodels.OptimizersConfigDiff(
            default_segment_number=2,
        ),
        replication_factor=int(os.getenv("QDRANT_RF", "1")),
    )
    # Helpful payload indexes
    for key, field_type in [
        ("metadata.filename", qmodels.PayloadSchemaType.KEYWORD),
        ("metadata.index", qmodels.PayloadSchemaType.KEYWORD),
        ("metadata.title", qmodels.PayloadSchemaType.KEYWORD),
    ]:
        try:
            qdrant.create_payload_index(
                collection_name=name,
                field_name=key,
                field_schema=field_type,
            )
        except Exception:
            pass

def sha_id(*parts: str) -> int:
    """Deterministic numeric ID for Qdrant (64-bit slice of SHA-256)."""
    h = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts. Raises HTTPException on failure."""
    try:
        # Batch embeddings to reduce overhead; OpenAI supports batching via list input
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate_limit" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="OpenAI API rate limit exceeded during document processing. Please try again in a few moments."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embeddings: {error_msg}"
            )

# ------------------------------------------------------------------------------
# Pydantic models for request bodies
# ------------------------------------------------------------------------------
class UpdateSendableDescription(BaseModel):
    new_description: str

# ------------------------------------------------------------------------------
# Хелпер-функции
# ------------------------------------------------------------------------------
def get_company_doc_table(company_id: str):
    """Returns the collection name for the company's documents."""
    collection = company_doc_collection(company_id)
    ensure_collection(collection)
    return collection

def get_company_sendable_table(company_id: str):
    """Returns the collection name for the company's sendable files."""
    collection = company_sendable_collection(company_id)
    ensure_collection(collection)
    return collection

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

@app.delete("/{companyId}/fuckdrop") # для удаления всех коллекций компании
def clear_company_tables(companyId: str):
    """Clear existing collections for a company to fix schema issues."""
    doc_collection = company_doc_collection(companyId)
    sendable_collection = company_sendable_collection(companyId)
    
    try:
        # Delete document collection if it exists
        try:
            qdrant.delete_collection(doc_collection)
            print(f"Dropped collection '{doc_collection}'")
        except Exception:
            print(f"Collection '{doc_collection}' doesn't exist")
            
        # Delete sendable files collection if it exists
        try:
            qdrant.delete_collection(sendable_collection)
            print(f"Dropped collection '{sendable_collection}'")
        except Exception:
            print(f"Collection '{sendable_collection}' doesn't exist")
            
    except Exception as e:
        print(f"Error clearing collections: {str(e)}")

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
    filter_q = qmodels.Filter(
        must=[qmodels.FieldCondition(key="metadata.filename", match=qmodels.MatchValue(value=safe_filename))]
    )
    dup_count = qdrant.count(collection_name=table, count_filter=filter_q, exact=True).count
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

                # Store in Qdrant
                yield _sse_format("status", {"step": "store", "message": f"Storing {len(to_store)} chunks"})
                
                # Convert to Qdrant format
                # Include section title in the text we embed so titles influence search
                embed_inputs = []
                for item in to_store:
                    title = (item.get("metadata") or {}).get("title")
                    if title:
                        embed_inputs.append(f"Title: {title}\n\n{item['text']}")
                    else:
                        embed_inputs.append(item["text"])
                vectors = embed_texts(embed_inputs)
                points = []
                for i, (item, vector) in enumerate(zip(to_store, vectors)):
                    point_id = sha_id(companyId, safe_filename, str(i))
                    payload = {
                        "text": item["text"],
                        "metadata": item["metadata"]
                    }
                    points.append(qmodels.PointStruct(id=point_id, vector=vector, payload=payload))
                
                qdrant.upsert(collection_name=table, wait=True, points=points)
                total_count = qdrant.count(collection_name=table, exact=False).count
                
                yield _sse_format("done", {
                    "message": "Document processed and embeddings stored successfully.",
                    "row_count": int(total_count),
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

        # Convert to Qdrant format and store
        # Include section title in the text we embed so titles influence search
        embed_inputs = []
        for item in to_store:
            title = (item.get("metadata") or {}).get("title")
            if title:
                embed_inputs.append(f"Title: {title}\n\n{item['text']}")
            else:
                embed_inputs.append(item["text"])
        vectors = embed_texts(embed_inputs)
        points = []
        for i, (item, vector) in enumerate(zip(to_store, vectors)):
            point_id = sha_id(companyId, safe_filename, str(i))
            payload = {
                "text": item["text"],
                "metadata": item["metadata"]
            }
            points.append(qmodels.PointStruct(id=point_id, vector=vector, payload=payload))
        
        qdrant.upsert(collection_name=table, wait=True, points=points)
        total_count = qdrant.count(collection_name=table, exact=False).count
        
        response_obj = {
            "message": "Document processed and embeddings stored successfully.",
            "row_count": int(total_count),
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
    useReranker: bool = Query(True, description="Use reranker for better results"),
    rerankTopK: Optional[int] = Query(None, description="Number of docs to retrieve before reranking"),
    rerankerProvider: Optional[str] = Query(None, description="'openai' or 'jina' (overrides env flags)"),
):
    """
    Search through company documents using semantic search.
    Returns matching chunks with their metadata and relevance scores.
    """
    table = get_company_doc_table(companyId)

    # 1) Get embedding for the query
    try:
        emb_resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        query_vec = emb_resp.data[0].embedding
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate_limit" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="OpenAI API rate limit exceeded. Please try again in a few moments."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate query embedding: {error_msg}"
            )

    # 2) Determine reranker provider and search limits
    provider = (rerankerProvider or "").strip().lower() if rerankerProvider else None
    use_openai = False
    use_jina = False
    if useReranker:
        if provider == "openai":
            use_openai = OPENAI_RERANKER_ENABLED
        elif provider == "jina":
            use_jina = JINA_RERANKER_ENABLED and bool(JINA_API_KEY)
        else:
            # Default preference: OpenAI if enabled; otherwise Jina if enabled
            if OPENAI_RERANKER_ENABLED:
                use_openai = True
            elif JINA_RERANKER_ENABLED and bool(JINA_API_KEY):
                use_jina = True

    rerank_enabled = use_openai or use_jina
    search_limit = rerankTopK or (OPENAI_RERANKER_TOP_K if rerank_enabled else limit)
    
    # 3) Search using Qdrant (get more results if reranking)
    try:
        hits = qdrant.search(
            collection_name=table,
            query_vector=query_vec,
            limit=search_limit,
            with_payload=True,
            score_threshold=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    # 4) Format initial results
    initial_results = []
    for hit in hits:
        pl = hit.payload or {}
        md = pl.get("metadata", {})
        # Parse page numbers safely (stringified JSON, list, or None)
        pn_raw = md.get("page_numbers")
        page_numbers = None
        if isinstance(pn_raw, str):
            try:
                page_numbers = json.loads(pn_raw)
            except Exception:
                page_numbers = None
        elif isinstance(pn_raw, list):
            page_numbers = pn_raw

        initial_results.append({
            "text":         pl.get("text"),
            "filename":     md.get("filename"),
            "page_numbers": page_numbers,
            "title":        md.get("title"),
            "url":          md.get("url"),
            "score":        float(hit.score) if hit.score is not None else 0.0  # Vector similarity score
        })
    
    # 5) Apply reranking if enabled
    final_results = initial_results
    rerank_metadata = {"reranked": False}
    
    if rerank_enabled and len(initial_results) > 1:
        import time
        start_time = time.time()
        
        try:
            if use_openai:
                reranked_results = rerank_with_openai(
                    client=client,
                    query=query,
                    documents=initial_results,
                    top_k=limit,
                    enabled=True,
                    model=OPENAI_RERANKER_MODEL,
                    timeout=OPENAI_RERANKER_TIMEOUT,
                )
                rerank_metadata = {
                    "reranked": True,
                    "reranker_model": OPENAI_RERANKER_MODEL,
                    "reranker_provider": "openai",
                }
            else:
                reranked_results = rerank_with_jina(
                    query=query,
                    documents=initial_results,
                    top_k=limit,
                    api_key=JINA_API_KEY,
                    model=JINA_RERANKER_MODEL,
                    endpoint=JINA_RERANKER_ENDPOINT,
                    timeout=JINA_RERANKER_TIMEOUT,
                )
                rerank_metadata = {
                    "reranked": True,
                    "reranker_model": JINA_RERANKER_MODEL,
                    "reranker_provider": "jina",
                }
            if reranked_results:
                final_results = reranked_results
                rerank_metadata.update({
                    "original_count": len(initial_results),
                    "reranked_count": len(final_results),
                    "rerank_time_ms": int((time.time() - start_time) * 1000)
                })
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
    Delete a document and its chunks from both Qdrant and GCS.
    The filename parameter should be just the filename, not a JSON object.
    """
    # 1) Get the table
    table = get_company_doc_table(companyId)

    # 2) Count matching records and delete from Qdrant
    filter_q = qmodels.Filter(
        must=[qmodels.FieldCondition(key="metadata.filename", match=qmodels.MatchValue(value=filename))]
    )
    
    # Get count first
    count_result = qdrant.count(collection_name=table, count_filter=filter_q, exact=True)
    deleted = count_result.count

    # 3) Delete from Qdrant
    qdrant.delete(collection_name=table, points_selector=qmodels.FilterSelector(filter=filter_q))

    # 4) Delete from GCS
    path = f"uploads/{companyId}/documents/{filename}"
    try:
        bucket.blob(path).delete()
    except Exception as e:
        # Log the error but don't fail the request since the DB deletion succeeded
        print(f"Warning: Failed to delete file from GCS: {str(e)}")

    return {
        "message": f"Deleted {deleted} chunks from Qdrant and attempted to remove file from GCS.",
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

    # 1) Check for duplicates
    filter_q = qmodels.Filter(
        must=[qmodels.FieldCondition(key="metadata.filename", match=qmodels.MatchValue(value=safe_name))]
    )
    dup_count = qdrant.count(collection_name=table, count_filter=filter_q, exact=True).count
    if dup_count > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Sendable '{safe_name}' already processed for company {companyId}"
        )

    ### NEW: Generate a unique, sequential index for the sendable ###
    # 1.1) Calculate the next index by scanning existing indices
    indices = []
    cursor = None
    while True:
        batch, cursor = qdrant.scroll(
            collection_name=table, 
            with_payload=True, 
            limit=1000, 
            offset=cursor
        )
        for point in batch:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            index_str = metadata.get("index")
            if index_str and index_str.startswith("SF-"):
                try:
                    num = int(index_str.split('-')[1])
                    indices.append(num)
                except (ValueError, IndexError):
                    pass
        if cursor is None or not batch:
            break
    
    # The new index is the highest existing index + 1
    max_index = max(indices) if indices else 0
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
    try:
        emb_resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=description
        )
        embedding = emb_resp.data[0].embedding
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate_limit" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="OpenAI API rate limit exceeded. Please try again in a few moments."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embedding: {error_msg}"
            )

    # 4) Prepare record and save to Qdrant
    metadata = {
        "index": sendable_index,
        "file_type": (file.content_type or os.path.splitext(safe_name)[1]),
        "filename": safe_name,
        "url": gcs_url,
        "cloudinary_url": cloudinary_url,
        "cloudinary_public_id": cloudinary_public_id,
    }

    payload = {
        "text": description,
        "metadata": metadata
    }
    
    point_id = sha_id(companyId, "sendable", safe_name)
    point = qmodels.PointStruct(id=point_id, vector=embedding, payload=payload)
    qdrant.upsert(collection_name=table, points=[point], wait=True)

    ### MODIFIED ###
    total_count = qdrant.count(collection_name=table, exact=False).count
    return {
        "message":   "Sendable file processed and saved.",
        "row_count": int(total_count),
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

    # 1) Find the record to update using Qdrant
    filter_q = qmodels.Filter(
        must=[qmodels.FieldCondition(key="metadata.filename", match=qmodels.MatchValue(value=safe_name))]
    )
    found_records, _ = qdrant.scroll(
        collection_name=table, 
        scroll_filter=filter_q, 
        with_payload=True, 
        with_vectors=True,
        limit=2
    )
    
    if not found_records:
        raise HTTPException(
            status_code=404,
            detail=f"No sendable with filename '{safe_name}' found."
        )
    elif len(found_records) > 1:
        raise HTTPException(
            status_code=500,
            detail=f"Multiple sendables with filename '{safe_name}' found. Data integrity issue."
        )

    # 2) Recalculate embedding for the new description
    try:
        emb_resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=new_description
        )
        new_embedding = emb_resp.data[0].embedding
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate_limit" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="OpenAI API rate limit exceeded. Please try again in a few moments."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embedding: {error_msg}"
            )

    # 3) Update with new description and vector
    point = found_records[0]
    payload = point.payload or {}
    payload["text"] = new_description
    
    updated_point = qmodels.PointStruct(id=point.id, vector=new_embedding, payload=payload)
    qdrant.upsert(collection_name=table, points=[updated_point], wait=True)

    return {"message": f"Description for '{safe_name}' updated successfully."}

@app.get("/companies/{companyId}/search-sendable")
async def search_sendable(
    companyId: str,
    query:     str   = Query(...),
    limit:     int   = Query(5)
):
    table = get_company_sendable_table(companyId)

    # 1) Получаем embedding для запроса
    try:
        emb_resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        query_vec = emb_resp.data[0].embedding
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate_limit" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="OpenAI API rate limit exceeded. Please try again in a few moments."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate query embedding: {error_msg}"
            )

    # 2) Search using Qdrant
    hits = qdrant.search(
        collection_name=table,
        query_vector=query_vec,
        limit=limit,
        with_payload=True,
    )

    # 3) Format results
    results = []
    for hit in hits:
        pl = hit.payload or {}
        md = pl.get("metadata", {})
        results.append({
            "index":     md.get("index"),
            "text":      pl.get("text"),
            "filename":  md.get("filename"),
            "file_type": md.get("file_type"),
            "url":       md.get("url"),
            "cloudinary_url": md.get("cloudinary_url"),
            "score":     float(hit.score) if hit.score is not None else 0.0
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

    # 2) Execute the search using Qdrant filter
    filter_q = qmodels.Filter(
        must=[qmodels.FieldCondition(key="metadata.index", match=qmodels.MatchValue(value=query))]
    )
    found_records, _ = qdrant.scroll(
        collection_name=table, 
        scroll_filter=filter_q, 
        with_payload=True, 
        limit=limit
    )

    # 3) Check if a record was found.
    if not found_records:
        raise HTTPException(
            status_code=404,
            detail=f"No sendable file found with index '{query}' for company {companyId}."
        )

    # 4) Format and return the single result.
    point = found_records[0]
    payload = point.payload or {}
    metadata = payload.get("metadata", {})
    
    # We construct a response object similar to the vector search for consistency.
    result = {
        "index":     metadata.get("index"),
        "text":      payload.get("text"),
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
        filter_q = qmodels.Filter(
            must=[qmodels.FieldCondition(key="metadata.filename", match=qmodels.MatchValue(value=safe_name))]
        )
        identifier = f"filename '{safe_name}'"
    else: # We know index is not None here because of the validation above
        filter_q = qmodels.Filter(
            must=[qmodels.FieldCondition(key="metadata.index", match=qmodels.MatchValue(value=index))]
        )
        identifier = f"index '{index}'"

    # 2) Find the record(s) to delete *before* deleting from the DB
    # We need the metadata to clean up GCS and Cloudinary
    records_to_delete, _ = qdrant.scroll(
        collection_name=table, 
        scroll_filter=filter_q, 
        with_payload=True, 
        limit=10000
    )
    
    deleted_count = len(records_to_delete)
    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No sendable found with {identifier} for deletion."
        )

    # 3) Delete associated cloud assets (GCS and Cloudinary)
    for point in records_to_delete:
        payload = point.payload or {}
        metadata = payload.get("metadata", {})
        
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

    # 4) Delete from Qdrant (now that cloud assets are gone)
    qdrant.delete(collection_name=table, points_selector=qmodels.FilterSelector(filter=filter_q))

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
    Берём файл, прогоняем через Docling, возвращаем текст. Не храним в Qdrant.
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
    Returns a list of documents for a company from Qdrant,
    including each file's name and metadata.
    """
    # 1) Get the collection for this company
    collection = get_company_doc_table(companyId)
    
    # 2) Scroll through all documents in Qdrant
    documents = {}
    cursor = None
    while True:
        batch, cursor = qdrant.scroll(
            collection_name=collection, 
            with_payload=True, 
            limit=1000, 
            offset=cursor
        )
        if not batch:
            break
        for point in batch:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            filename = metadata.get("filename")
            if filename and filename not in documents:
                documents[filename] = {
                    "filename": filename,
                    "title": metadata.get("title"),
                    "url": metadata.get("url")
                }
        if cursor is None:
            break
    
    # 4) Convert to list
    results = list(documents.values())
    
    return {"files": results}

@app.get("/companies/{companyId}/sendables/list")
async def list_sendables(companyId: str):
    collection = get_company_sendable_table(companyId)
    
    results = []
    cursor = None
    while True:
        batch, cursor = qdrant.scroll(
            collection_name=collection, 
            with_payload=True, 
            limit=1000, 
            offset=cursor
        )
        if not batch:
            break
        for point in batch:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            result_item = {
                "index": metadata.get("index"),
                "filename": metadata.get("filename"),
                "text": payload.get("text"),
                "file_type": metadata.get("file_type"),
                "url": metadata.get("url"),
                "cloudinary_url": metadata.get("cloudinary_url")
            }
            results.append(result_item)
        if cursor is None:
            break
    
    return {"sendables": results}

@app.get("/companies/{companyId}/documents/content")
async def get_all_document_content(companyId: str):
    """
    Returns the content of all documents for a company, including their text and metadata.
    This is useful for bulk processing or AI summarization of the entire document base.
    """
    collection = get_company_doc_table(companyId)
    
    # Get all documents from Qdrant
    documents = {}
    cursor = None
    while True:
        batch, cursor = qdrant.scroll(
            collection_name=collection, 
            with_payload=True, 
            limit=1000, 
            offset=cursor
        )
        if not batch:
            break
        for point in batch:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            filename = metadata.get("filename")
            if not filename:
                continue
            if filename not in documents:
                documents[filename] = {
                    "filename": filename,
                    "title": metadata.get("title"),
                    "url": metadata.get("url"),
                    "chunks": []
                }
            documents[filename]["chunks"].append({
                "text": payload.get("text", ""),
                "page_numbers": metadata.get("page_numbers")
            })
        if cursor is None:
            break
    
    # Convert to list and sort chunks by page numbers
    result = []
    for doc in documents.values():
        # Sort chunks by page numbers if available
        first_pn = doc["chunks"][0].get("page_numbers") if doc["chunks"] else None
        if first_pn:
            def _min_page(x):
                pn = x.get("page_numbers")
                if isinstance(pn, str):
                    try:
                        pn = json.loads(pn)
                    except Exception:
                        pn = None
                if isinstance(pn, list) and pn:
                    return min(pn)
                return float('inf')
            doc["chunks"].sort(key=_min_page)
        
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



# utils/processing.py
import re
import os
import json
from typing import List, Dict, Any, Optional
import urllib.parse

# Импортируем Docling, OpenAI и другие нужные вещи

from utils.tokenizer import OpenAITokenizerWrapper # Предполагая, что у вас есть этот файл
from config import client, bucket, tokenizer # Настройки из config

# Инициализация здесь

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
    """Format data as Server-Sent Event with proper encoding handling."""
    try:
        payload = json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        # Fallback for non-serializable objects
        try:
            payload = json.dumps({"message": str(data)}, ensure_ascii=False)
        except Exception:
            payload = json.dumps({"message": repr(data), "error": "encoding_issue"})
    except Exception as e:
        payload = json.dumps({"error": "serialization_failed", "details": repr(e)})
    return f"event: {event}\ndata: {payload}\n\n"

## Reranker moved to utils/reranker.py

# ------------------------------------------------------------------------------
# Markdown-based section/paragraph extraction (Code-Block Safe)
# ------------------------------------------------------------------------------
def _extract_sections_from_markdown(md: str) -> List[Dict[str, Any]]:
    """
    Parse markdown into sections and paragraphs. IGNORES headers inside code blocks.
    Returns a list of sections:
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
    
    # Flag: are we inside a ``` code block?
    in_code_block = False 

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
        stripped_ln = ln.strip()
        
        # 1. Detect code block boundaries
        if stripped_ln.startswith("```"):
            in_code_block = not in_code_block
            # Code block is part of the current paragraph
            if section_idx < 0: _start_new_section(None, 1)
            current_paragraph_lines.append(ln)
            continue

        # 2. If inside code block, IGNORE header detection
        if in_code_block:
            if section_idx < 0: _start_new_section(None, 1)
            current_paragraph_lines.append(ln)
            continue

        # 3. Heading detection (ONLY if not in code)
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            _flush_paragraph()
            current_title = m.group(2).strip()
            current_level = len(m.group(1))
            _start_new_section(current_title, current_level)
            continue

        # 4. Blank line => paragraph boundary
        if not ln.strip():
            _flush_paragraph()
            continue

        # 5. Normal text
        if section_idx < 0:
            _start_new_section(title=None, level=1)
        current_paragraph_lines.append(ln)

    _flush_paragraph()

    if not sections and md.strip():
        sections = [{
            "title": None, "level": 1, 
            "paragraphs": [{"text": md.strip(), "para_idx": 0}], 
            "section_idx": 0
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
# Safe Chunk Generator (for LLM Windowing)
# ------------------------------------------------------------------------------
from typing import Iterator
import hashlib
from pydantic import BaseModel

def safe_chunk_generator(text: str, chunk_size: int = 3000) -> Iterator[str]:
    """
    Yields chunks of text, splitting only at paragraph boundaries (\n\n).
    Ensures LLM inputs are always complete paragraphs.
    """
    if not text:
        return
    
    remainder = ""
    pos = 0
    text_len = len(text)
    
    while pos < text_len:
        end_pos = min(pos + chunk_size, text_len)
        chunk = remainder + text[pos:end_pos]
        
        if end_pos >= text_len:
            if chunk.strip():
                yield chunk
            break
        
        last_break = chunk.rfind("\n\n")
        
        if last_break > 0:
            yield chunk[:last_break]
            remainder = chunk[last_break:].lstrip("\n")
        else:
            yield chunk
            remainder = ""
        
        pos = end_pos

# ------------------------------------------------------------------------------
# Recursive text splitter for large sections
# ------------------------------------------------------------------------------
def _recursive_split_text(text: str, max_tokens: int) -> List[str]:
    """Recursively splits text by \n\n, then \n, then '. ' to respect token limits."""
    if _count_tokens(text) <= max_tokens:
        return [text]
    
    parts = text.split("\n\n")
    if len(parts) > 1:
        return _split_and_merge(parts, max_tokens, "\n\n")
    
    parts = text.split("\n")
    if len(parts) > 1:
        return _split_and_merge(parts, max_tokens, "\n")
    
    parts = text.split(". ")
    if len(parts) > 1:
        parts = [p + "." if i < len(parts) - 1 else p for i, p in enumerate(parts)]
        return _split_and_merge(parts, max_tokens, " ")
    
    return [text]

def _split_and_merge(parts: List[str], max_tokens: int, separator: str) -> List[str]:
    """Merge parts back together respecting max_tokens."""
    result = []
    current_chunk = ""
    
    for part in parts:
        test_chunk = current_chunk + separator + part if current_chunk else part
        if _count_tokens(test_chunk) <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                result.append(current_chunk)
            if _count_tokens(part) > max_tokens:
                result.extend(_recursive_split_text(part, max_tokens))
                current_chunk = ""
            else:
                current_chunk = part
    
    if current_chunk:
        result.append(current_chunk)
    
    return result

# ------------------------------------------------------------------------------
# Semantic Parent-Child Chunking
# ------------------------------------------------------------------------------
def _build_semantic_chunks(
    sections: List[Dict[str, Any]],
    max_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """
    Build chunks using Parent-Child strategy:
    - Parent = Full section text
    - If section <= max_tokens: 1 chunk (child = parent)
    - If section > max_tokens: Split into children, all referencing same parent
    
    Returns list of chunks with parent_id and parent_text for Small-to-Big retrieval.
    """
    chunks: List[Dict[str, Any]] = []

    for section in sections:
        section_title = section.get("title")
        parent_text = "\n\n".join(p.get("text", "") for p in section.get("paragraphs", []))
        if not parent_text.strip():
            continue
        
        parent_id = hashlib.md5(f"{section_title or ''}:{parent_text}".encode()).hexdigest()[:16]
        parent_tokens = _count_tokens(parent_text)
        
        if parent_tokens <= max_tokens:
            chunks.append({
                "text": parent_text,
                "section_title": section_title,
                "parent_id": parent_id,
                "parent_text": parent_text,
            })
        else:
            child_texts = _recursive_split_text(parent_text, max_tokens)
            for child_text in child_texts:
                chunks.append({
                    "text": child_text,
                    "section_title": section_title,
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                })
    
    return chunks

# ------------------------------------------------------------------------------
# LLM Split-Stitch Enhancement
# ------------------------------------------------------------------------------
class SplitPoint(BaseModel):
    start_snippet: str  # 15-20 words verbatim from text
    title: str          # Generated markdown header

def _identify_split_points_llm(chunk_text: str) -> List[SplitPoint]:
    """Call LLM to identify semantic topic shifts in a chunk."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a document structure analyst. Find semantic topic shifts in the text. "
                    "For each new logical section, provide: "
                    "1. start_snippet: The FIRST 15-20 words of the sentence where the new section starts (VERBATIM copy). "
                    "2. title: A markdown header title (e.g., '## Cancellation Policy'). "
                    "Ignore hashtags, code comments, existing headers. Return JSON array."
                )},
                {"role": "user", "content": f"Find topic shifts in this text:\n\n{chunk_text}"},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content if resp.choices else None
        if not content:
            return []
        
        data = json.loads(content)
        points = data.get("split_points", data.get("sections", []))
        return [SplitPoint(start_snippet=p.get("start_snippet", ""), title=p.get("title", "")) 
                for p in points if p.get("start_snippet")]
    except Exception as e:
        print(f"[LLM Split] Error: {e}")
        return []

def _stitch_text_with_headers(raw_text: str, split_points: List[SplitPoint]) -> str:
    """Insert headers into raw_text at positions identified by split_points."""
    if not split_points:
        return raw_text
    
    result_parts = []
    cursor = 0
    
    for point in split_points:
        if not point.start_snippet:
            continue
        
        found_idx = raw_text.find(point.start_snippet, cursor)
        if found_idx == -1:
            partial = point.start_snippet[:50]
            found_idx = raw_text.find(partial, cursor)
        
        if found_idx != -1:
            result_parts.append(raw_text[cursor:found_idx])
            result_parts.append(f"\n\n{point.title}\n\n")
            cursor = found_idx
    
    result_parts.append(raw_text[cursor:])
    return "".join(result_parts)

def _enhance_text_with_llm(text: str, chunk_size: int = 3000) -> str:
    """Enhance raw markdown by identifying and inserting semantic headers."""
    all_split_points: List[SplitPoint] = []
    
    for chunk in safe_chunk_generator(text, chunk_size):
        points = _identify_split_points_llm(chunk)
        all_split_points.extend(points)
    
    if not all_split_points:
        return text
    
    return _stitch_text_with_headers(text, all_split_points)

# ------------------------------------------------------------------------------
# Main Pipeline Wrapper
# ------------------------------------------------------------------------------
def run_smart_chunking_pipeline(
    text: str,
    llm_enhance: bool = True,
    max_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """
    Full Smart Chunking pipeline:
    1. (Optional) Enhance text with LLM-identified headers
    2. Extract sections using code-block-safe parser
    3. Build semantic chunks with Parent-Child structure
    
    Returns list of chunks ready for indexing, each containing:
    - text: The chunk text (for embedding)
    - section_title: Header/title of the section
    - parent_id: Unique ID for the parent section (for deduplication)
    - parent_text: Full parent section text (for Small-to-Big retrieval)
    """
    enhanced_text = text
    if llm_enhance:
        try:
            enhanced_text = _enhance_text_with_llm(text)
        except Exception as e:
            print(f"[Smart Chunking] LLM enhancement failed, using original text: {e}")
    
    sections = _extract_sections_from_markdown(enhanced_text)
    chunks = _build_semantic_chunks(sections, max_tokens)
    
    return chunks

def safe_decode_filename(filename: str) -> str:
    """Fixes improperly decoded filenames with robust error handling."""
    try:
        # First try URL decoding in case the filename is URL-encoded
        decoded = urllib.parse.unquote(filename)
        # Then try to encode/decode to handle any remaining encoding issues
        return decoded.encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        # If any encoding/decoding fails, return the URL-decoded version
        try:
            return urllib.parse.unquote(filename)
        except Exception:
            # Last resort: return as-is
            return filename
    except Exception:
        # Catch-all: return filename as-is
        return filename

def upload_to_gcs(content: bytes, company_id: str, folder: str, filename: str, content_type: str) -> str:
    """Uploads content to Google Cloud Storage."""
    path = f"uploads/{company_id}/{folder}/{filename}"
    blob = bucket.blob(path)
    blob.upload_from_string(content, content_type=content_type)
    return f"gs://{bucket.name}/{path}"

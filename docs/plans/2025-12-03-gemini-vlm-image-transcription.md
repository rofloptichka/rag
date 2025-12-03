# Gemini VLM Image Transcription Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable automatic image transcription in documents using Gemini 2.0 Flash via Docling's built-in VLM API integration, with fallback to GPT-4o-mini on failure.

**Architecture:** Configure Docling's `PictureDescriptionApiOptions` to use Gemini's OpenAI-compatible endpoint. Image descriptions are inserted inline where images appear, then processed by existing chunking pipeline. Retry logic with exponential backoff handles transient failures; GPT-4o-mini serves as fallback provider.

**Tech Stack:** Docling 2.58.0+, Gemini 2.0 Flash API, OpenAI GPT-4o-mini (fallback), existing FastAPI backend

---

## Task 1: Add VLM Configuration to config.py

**Files:**
- Modify: `spongeragprod/rag/config.py:14-16` (imports)
- Modify: `spongeragprod/rag/config.py:75-79` (converter initialization)

**Step 1: Add the new imports**

Add after line 16 (`import google.generativeai as genai`):

```python
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
```

**Step 2: Add VLM configuration constants**

Add after line 39 (after `EMBED_TASK_TYPE`):

```python
# Gemini VLM for image transcription
GEMINI_VLM_MODEL = os.getenv("GEMINI_VLM_MODEL", "gemini-2.0-flash")
GEMINI_VLM_TIMEOUT = int(os.getenv("GEMINI_VLM_TIMEOUT", "30"))
GEMINI_VLM_PROMPT = """Analyze this image and provide:
1. OCR: Extract all visible text exactly as written
2. Description: Describe what the image shows (charts, diagrams, photos, etc.)
3. Summary: One sentence explaining the image's purpose or meaning

Format your response as:
[TEXT]: <extracted text or "No text visible">
[DESCRIPTION]: <what the image contains>
[SUMMARY]: <purpose/meaning>"""

# Fallback VLM (OpenAI) for when Gemini fails
OPENAI_VLM_MODEL = os.getenv("OPENAI_VLM_MODEL", "gpt-4o-mini")
OPENAI_VLM_TIMEOUT = int(os.getenv("OPENAI_VLM_TIMEOUT", "30"))
```

**Step 3: Replace the converter initialization**

Replace lines 75-79:

```python
# Docling configuration
converter = DocumentConverter()
tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = int(os.getenv("HYBRID_CHUNK_MAX_TOKENS", "512"))
chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)
```

With:

```python
# Docling configuration with Gemini VLM for image transcription
tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = int(os.getenv("HYBRID_CHUNK_MAX_TOKENS", "512"))
chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)

# Configure PDF pipeline with Gemini VLM for picture descriptions
_pdf_pipeline_options = PdfPipelineOptions(
    enable_remote_services=True,
    do_picture_description=True,
)
_pdf_pipeline_options.picture_description_options = PictureDescriptionApiOptions(
    url=f"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    headers={"Authorization": f"Bearer {GOOGLE_API_KEY}"},
    params={
        "model": GEMINI_VLM_MODEL,
        "max_tokens": 500,
    },
    prompt=GEMINI_VLM_PROMPT,
    timeout=GEMINI_VLM_TIMEOUT,
    provenance="gemini-2.0-flash",
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_pipeline_options),
    }
)
```

**Step 4: Verify the changes compile**

Run:
```bash
cd /mnt/c/users/bfrey/sponge/spongeragprod/rag && python -c "from config import converter; print('OK')"
```

Expected: `OK` (no import errors)

**Step 5: Commit**

```bash
git add spongeragprod/rag/config.py
git commit -m "feat: add Gemini VLM configuration for image transcription in Docling"
```

---

## Task 2: Create VLM Retry Wrapper with Fallback

**Files:**
- Create: `spongeragprod/rag/utils/vlm_converter.py`

**Step 1: Create the VLM converter wrapper**

Create new file `spongeragprod/rag/utils/vlm_converter.py`:

```python
"""
VLM-enabled document converter with retry and fallback logic.
Wraps Docling's DocumentConverter to handle Gemini API failures gracefully.
"""
import logging
import time
from pathlib import Path
from typing import Union

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from config import (
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    GEMINI_VLM_MODEL,
    GEMINI_VLM_TIMEOUT,
    GEMINI_VLM_PROMPT,
    OPENAI_VLM_MODEL,
    OPENAI_VLM_TIMEOUT,
)

_log = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
BACKOFF_MULTIPLIER = 2.0


def _create_gemini_converter() -> DocumentConverter:
    """Create a DocumentConverter configured with Gemini VLM."""
    pipeline_options = PdfPipelineOptions(
        enable_remote_services=True,
        do_picture_description=True,
    )
    pipeline_options.picture_description_options = PictureDescriptionApiOptions(
        url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        headers={"Authorization": f"Bearer {GOOGLE_API_KEY}"},
        params={"model": GEMINI_VLM_MODEL, "max_tokens": 500},
        prompt=GEMINI_VLM_PROMPT,
        timeout=GEMINI_VLM_TIMEOUT,
        provenance="gemini-2.0-flash",
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _create_openai_converter() -> DocumentConverter:
    """Create a DocumentConverter configured with OpenAI VLM (fallback)."""
    pipeline_options = PdfPipelineOptions(
        enable_remote_services=True,
        do_picture_description=True,
    )
    pipeline_options.picture_description_options = PictureDescriptionApiOptions(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        params={"model": OPENAI_VLM_MODEL, "max_tokens": 500},
        prompt=GEMINI_VLM_PROMPT,
        timeout=OPENAI_VLM_TIMEOUT,
        provenance="gpt-4o-mini",
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _create_no_vlm_converter() -> DocumentConverter:
    """Create a DocumentConverter without VLM (final fallback)."""
    return DocumentConverter()


def convert_with_vlm_fallback(source: Union[str, Path]):
    """
    Convert a document with VLM image transcription.

    Tries Gemini first with exponential backoff retries.
    Falls back to OpenAI GPT-4o-mini if Gemini fails.
    Falls back to no VLM if both fail (images will be placeholders).

    Args:
        source: Path to the document file

    Returns:
        ConversionResult from Docling
    """
    # Try Gemini with retries
    gemini_converter = _create_gemini_converter()
    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES):
        try:
            _log.info(f"Converting with Gemini VLM (attempt {attempt + 1}/{MAX_RETRIES})")
            result = gemini_converter.convert(source=source)
            _log.info("Gemini VLM conversion successful")
            return result
        except Exception as e:
            _log.warning(f"Gemini VLM failed (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                _log.info(f"Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER

    # Try OpenAI fallback
    try:
        _log.info("Falling back to OpenAI GPT-4o-mini VLM")
        openai_converter = _create_openai_converter()
        result = openai_converter.convert(source=source)
        _log.info("OpenAI VLM conversion successful")
        return result
    except Exception as e:
        _log.warning(f"OpenAI VLM fallback failed: {e}")

    # Final fallback: no VLM
    _log.warning("All VLM providers failed, converting without image transcription")
    no_vlm_converter = _create_no_vlm_converter()
    return no_vlm_converter.convert(source=source)
```

**Step 2: Verify the module imports correctly**

Run:
```bash
cd /mnt/c/users/bfrey/sponge/spongeragprod/rag && python -c "from utils.vlm_converter import convert_with_vlm_fallback; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add spongeragprod/rag/utils/vlm_converter.py
git commit -m "feat: add VLM converter wrapper with retry and fallback logic"
```

---

## Task 3: Update Document Router to Use VLM Converter

**Files:**
- Modify: `spongeragprod/rag/routers/documents.py:145` (SSE endpoint)
- Modify: `spongeragprod/rag/routers/documents.py:312` (sync endpoint)

**Step 1: Add the import**

At the top of `documents.py`, add:

```python
from utils.vlm_converter import convert_with_vlm_fallback
```

**Step 2: Update SSE endpoint (line 145)**

Replace:
```python
result = converter.convert(source=temp_path)
```

With:
```python
result = convert_with_vlm_fallback(source=temp_path)
```

**Step 3: Update sync endpoint (line 312)**

Replace:
```python
result = converter.convert(source=temp_path)
```

With:
```python
result = convert_with_vlm_fallback(source=temp_path)
```

**Step 4: Verify no syntax errors**

Run:
```bash
cd /mnt/c/users/bfrey/sponge/spongeragprod/rag && python -c "from routers.documents import router; print('OK')"
```

Expected: `OK`

**Step 5: Commit**

```bash
git add spongeragprod/rag/routers/documents.py
git commit -m "feat: use VLM converter with fallback for document processing"
```

---

## Task 4: Add Environment Variables to .env.example

**Files:**
- Modify: `spongeragprod/rag/.env.example` (or create if doesn't exist)

**Step 1: Add VLM configuration section**

Add to `.env.example`:

```bash
# === VLM Image Transcription ===
# Gemini VLM (primary)
GEMINI_VLM_MODEL=gemini-2.0-flash
GEMINI_VLM_TIMEOUT=30

# OpenAI VLM (fallback)
OPENAI_VLM_MODEL=gpt-4o-mini
OPENAI_VLM_TIMEOUT=30
```

**Step 2: Commit**

```bash
git add spongeragprod/rag/.env.example
git commit -m "docs: add VLM configuration to .env.example"
```

---

## Task 5: Manual Integration Test

**Step 1: Prepare a test PDF with images**

Find or create a simple PDF with at least one image (chart, diagram, or photo).

**Step 2: Start the server**

```bash
cd /mnt/c/users/bfrey/sponge/spongeragprod/rag
uvicorn main:app --reload
```

**Step 3: Upload the test document**

Use your existing upload endpoint. Watch the server logs for:
- `Converting with Gemini VLM (attempt 1/3)`
- `Gemini VLM conversion successful`

**Step 4: Verify image content in chunks**

Check the resulting chunks in Qdrant. Image descriptions should appear inline with surrounding text, containing `[TEXT]:`, `[DESCRIPTION]:`, and `[SUMMARY]:` sections.

**Step 5: Test fallback (optional)**

Temporarily set an invalid `GOOGLE_API_KEY` to trigger fallback:
- Should see retry attempts with exponential backoff
- Should see `Falling back to OpenAI GPT-4o-mini VLM`
- Document should still process successfully

---

## Summary of Changes

| File | Change |
|------|--------|
| `config.py` | Add VLM imports, constants, and pipeline configuration |
| `utils/vlm_converter.py` | New file: retry wrapper with Gemini → OpenAI → no-VLM fallback |
| `routers/documents.py` | Replace `converter.convert()` with `convert_with_vlm_fallback()` |
| `.env.example` | Document new environment variables |

**Total: ~100 lines of new code, 2 line changes in existing code.**

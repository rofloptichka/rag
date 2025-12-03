"""
VLM-enabled document converter with retry and fallback logic.
Wraps Docling's DocumentConverter to handle Gemini API failures gracefully.
"""
import logging
import os
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

# Rate limiting: delay between VLM API calls (in seconds)
VLM_REQUEST_DELAY = float(os.getenv("VLM_REQUEST_DELAY", "0.5"))

# Monkey-patch Docling's api_image_request to add rate limiting
def _patch_api_request_with_delay():
    """Add delay between API calls to avoid rate limits."""
    try:
        from docling.utils import api_image_request as api_module
        original_func = api_module.api_image_request

        def rate_limited_api_image_request(*args, **kwargs):
            if VLM_REQUEST_DELAY > 0:
                time.sleep(VLM_REQUEST_DELAY)
            return original_func(*args, **kwargs)

        api_module.api_image_request = rate_limited_api_image_request
        _log.info(f"VLM rate limiting enabled: {VLM_REQUEST_DELAY}s delay between requests")
    except Exception as e:
        _log.warning(f"Could not patch api_image_request for rate limiting: {e}")

# Apply the patch on module load
_patch_api_request_with_delay()


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
        picture_area_threshold=0.01,
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
        picture_area_threshold=0.01,
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

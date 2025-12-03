import unicodedata
import regex
import math
import logging
import time
from config import r  # Импортируем готовый клиент Redis из config.py
from typing import Dict, Any
from functools import lru_cache


logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# BM25 Stats Cache (reduces Redis round-trips during search)
# ------------------------------------------------------------------------------
_bm25_stats_cache: Dict[str, Dict[str, Any]] = {}
_bm25_cache_ttl: Dict[str, float] = {}
_BM25_CACHE_TTL_SECONDS = 30  # Cache stats for 30 seconds

# ------------------------------------------------------------------------------
# Multilingual Tokenization & Normalization Utilities
# ------------------------------------------------------------------------------
# Language-aware tokenization for CJK (Chinese, Japanese, Korean) and other scripts
_RE_CJK = regex.compile(r'[\p{Han}\p{Katakana}\p{Hiragana}\p{Hangul}]')
_RE_WORD = regex.compile(r'[\p{L}\p{N}]+', flags=regex.UNICODE)
_STOP_LATIN = set("""a an the and or not of to in on for with at by from is are was were be been being as that this those these it its they them he she we you your our their i""".split())

def _normalize_sparse(t: str) -> str:
    """Normalize text using NFKC normalization and convert to lowercase."""
    return unicodedata.normalize('NFKC', t).lower()

def _tokenize_multilingual(text: str) -> list[str]:
    """
    Tokenize text with language-aware handling:
    - CJK characters: unigrams + bigrams
    - Other scripts: word tokens (2+ chars, excluding common stop words)
    """
    t = _normalize_sparse(text)
    toks: list[str] = []
    i = 0
    L = len(t)
    while i < L:
        ch = t[i]
        if _RE_CJK.match(ch):
            # CJK region: collect consecutive CJK chars
            j = i
            buf = []
            while j < L and _RE_CJK.match(t[j]):
                buf.append(t[j])
                j += 1
            s = ''.join(buf)
            toks.extend(list(s))  # unigrams
            toks.extend([s[k:k+2] for k in range(len(s)-1)])  # bigrams
            i = j
        else:
            # Non-CJK: try to match a word
            m = _RE_WORD.match(t, i)
            if m:
                tok = m.group(0)
                if len(tok) >= 2 or tok.isdigit() and tok not in _STOP_LATIN:
                    toks.append(tok)
                i = m.end()
            else:
                i += 1
    return toks

def _bm25_keys(coll: str) -> Dict[str, str]:
    """Generate Redis keys for BM25 statistics for a given collection."""
    return {
        "N": f"bm25:{coll}:N",        # Total number of documents
        "AVGDL": f"bm25:{coll}:AVGDL",  # Average document length
        "DF": f"bm25:{coll}:DF",       # Document frequency hash
    }

def bm25_update_stats_redis(coll: str, tokens: list[str]) -> None:
    """
    Update BM25 statistics in Redis for the given collection.
    
    Args:
        coll: Collection name
        tokens: List of tokens from the document
    """
    if r is None:
        logger.debug("Redis not available, skipping BM25 stats update")
        return
    
    try:
        keys = _bm25_keys(coll)
        pipe = r.pipeline(transaction=False)
        
        # Increment total document count
        pipe.incr(keys["N"])
        
        # Update average document length using exponential moving average
        avgdl_old = float(r.get(keys["AVGDL"]) or 200.0)
        new_avg = 0.99 * avgdl_old + 0.01 * len(tokens)
        pipe.set(keys["AVGDL"], new_avg)
        
        # Update document frequency for unique terms
        seen = set(tokens)
        for t in seen:
            pipe.hincrby(keys["DF"], t, 1)
        
        pipe.execute()
        logger.debug(f"BM25 stats updated for collection '{coll}', doc_length={len(tokens)}")
    except Exception as e:
        logger.warning(f"Failed to update BM25 stats in Redis: {e}")

def _idf_redis(term: str, coll: str, eps: float = 0.5) -> float:
    """
    Calculate IDF (Inverse Document Frequency) for a term using Redis stats.
    
    Args:
        term: The term to calculate IDF for
        coll: Collection name
        eps: Smoothing parameter (default: 0.5)
    
    Returns:
        IDF score for the term
    """
    if r is None:
        return 1.0  # Fallback when Redis is unavailable
    
    try:
        keys = _bm25_keys(coll)
        N = int(r.get(keys["N"]) or 1)
        df = int(r.hget(keys["DF"], term) or 1)
        return math.log(1 + (N - df + eps) / (df + eps))
    except Exception as e:
        logger.warning(f"Failed to get IDF from Redis: {e}")
        return 1.0  # Fallback value

def _get_stats(coll: str) -> Dict[str, Any]:
    """
    Get BM25 statistics for a collection.
    
    Args:
        coll: Collection name
    
    Returns:
        Dictionary containing N (total docs), AVGDL (average doc length), and DF (document frequencies)
    """
    if r is None:
        return {"N": 1, "AVGDL": 200.0, "DF": {}}
    
    try:
        keys = _bm25_keys(coll)
        N = int(r.get(keys["N"]) or 1)
        AVGDL = float(r.get(keys["AVGDL"]) or 200.0)
        # DF is a hash, we'll access it per-term when needed
        return {"N": N, "AVGDL": AVGDL, "DF": keys["DF"]}
    except Exception as e:
        logger.warning(f"Failed to get BM25 stats from Redis: {e}")
        return {"N": 1, "AVGDL": 200.0, "DF": {}}

def _get_stats_cached(coll: str) -> Dict[str, Any]:
    """
    Get BM25 statistics with in-memory caching to reduce Redis round-trips.
    Cached for 30 seconds - suitable for search queries where stats don't change frequently.
    
    Args:
        coll: Collection name
    
    Returns:
        Dictionary containing N (total docs), AVGDL (average doc length), and DF key
    """
    now = time.time()
    cached_time = _bm25_cache_ttl.get(coll, 0)
    
    if coll in _bm25_stats_cache and (now - cached_time) < _BM25_CACHE_TTL_SECONDS:
        return _bm25_stats_cache[coll]
    
    # Fetch fresh stats
    stats = _get_stats(coll)
    _bm25_stats_cache[coll] = stats
    _bm25_cache_ttl[coll] = now
    return stats

def _idf(term: str, N: int, DF_key: str, eps: float = 0.5) -> float:
    """
    Calculate IDF using provided stats.
    
    Args:
        term: The term to calculate IDF for
        N: Total number of documents
        DF_key: Redis hash key for document frequencies
        eps: Smoothing parameter
    
    Returns:
        IDF score
    """
    if r is None:
        return 1.0
    
    try:
        df = int(r.hget(DF_key, term) or 1)
        return math.log(1 + (N - df + eps) / (df + eps))
    except Exception as e:
        logger.warning(f"Failed to calculate IDF: {e}")
        return 1.0

def bm25_update_stats(coll: str, tokens: list[str]) -> None:
    """
    Alias for bm25_update_stats_redis for consistency with user's code.
    """
    bm25_update_stats_redis(coll, tokens)


def bm25_decrement_stats(coll: str, tokens: list[str]) -> None:
    """
    Decrement BM25 statistics in Redis when a document is deleted/replaced.
    This is the inverse of bm25_update_stats to prevent stat inflation on resyncs.

    Args:
        coll: Collection name
        tokens: List of tokens from the document being removed
    """
    if r is None:
        logger.debug("Redis not available, skipping BM25 stats decrement")
        return

    if not tokens:
        return

    try:
        keys = _bm25_keys(coll)
        pipe = r.pipeline(transaction=False)

        # Decrement total document count (but not below 1)
        current_n = int(r.get(keys["N"]) or 1)
        if current_n > 1:
            pipe.decr(keys["N"])

        # Adjust average document length using reverse exponential moving average
        # This approximates removing the document's contribution to the average
        avgdl_old = float(r.get(keys["AVGDL"]) or 200.0)
        doc_len = len(tokens)
        # Reverse the EMA: if new_avg = 0.99 * old + 0.01 * doc_len
        # then removing: new_avg ≈ (old - 0.01 * doc_len) / 0.99
        # Simplified: just nudge it in the opposite direction
        new_avg = max(50.0, 0.99 * avgdl_old - 0.01 * doc_len + 0.01 * avgdl_old)
        pipe.set(keys["AVGDL"], new_avg)

        # Decrement document frequency for unique terms (but not below 1)
        seen = set(tokens)
        for t in seen:
            current_df = int(r.hget(keys["DF"], t) or 1)
            if current_df > 1:
                pipe.hincrby(keys["DF"], t, -1)
            elif current_df == 1:
                # Remove the term entirely if DF would become 0
                pipe.hdel(keys["DF"], t)

        pipe.execute()
        logger.debug(f"BM25 stats decremented for collection '{coll}', doc_length={doc_len}")
    except Exception as e:
        logger.warning(f"Failed to decrement BM25 stats in Redis: {e}")


def bm25_reset_stats(coll: str) -> None:
    """
    Reset all BM25 statistics for a collection.
    Used when reindexing a collection from scratch.

    Args:
        coll: Collection name
    """
    if r is None:
        logger.debug("Redis not available, skipping BM25 stats reset")
        return

    try:
        keys = _bm25_keys(coll)
        pipe = r.pipeline(transaction=False)

        # Delete all BM25 keys for this collection
        pipe.delete(keys["N"])
        pipe.delete(keys["AVGDL"])
        pipe.delete(keys["DF"])

        pipe.execute()

        # Clear cache for this collection
        if coll in _bm25_stats_cache:
            del _bm25_stats_cache[coll]
        if coll in _bm25_cache_ttl:
            del _bm25_cache_ttl[coll]

        logger.info(f"BM25 stats reset for collection '{coll}'")
    except Exception as e:
        logger.warning(f"Failed to reset BM25 stats in Redis: {e}")

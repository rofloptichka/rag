# utils/qdrant_helpers.py
import hashlib
import json
from collections import OrderedDict
from fastapi import HTTPException
from qdrant_client.http import models as qmodels
import os 
import regex
from collections import Counter
from typing import List, Optional, Dict, Any, Tuple
import asyncio
import google.generativeai as genai

# Импортируем клиентов и настройки из config.py
from config import qdrant, EMBED_MODEL, EMBED_DIMS, EMBED_TASK_TYPE

# Импортируем BM25 функции, которые нужны для создания sparse векторов
from .bm25_helpers import (
    bm25_update_stats, bm25_update_stats_redis, 
    _get_stats, _get_stats_cached, _idf, _tokenize_multilingual
)
import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Qdrant helper functions
# ------------------------------------------------------------------------------

def company_doc_collection(company_id: str, source: str = "general", agent_id: Optional[str] = None) -> str:
    """
    Get the collection name for company documents.

    Args:
        company_id: The company identifier
        source: Document source type - "general" for uploaded docs, "sheets" for Google Sheets/Docs
        agent_id: Optional agent identifier for per-agent isolation (required for "sheets" source)

    Returns:
        Collection name in format:
        - docling_{company_id} for general docs
        - docling_{company_id}_sheets_{agent_id} for agent-specific Google Sheets/Docs
    """
    if source == "general":
        return f"docling_{company_id}"
    # For source-specific collections (e.g., "sheets"), include agent_id for isolation
    if agent_id:
        return f"docling_{company_id}_{source}_{agent_id}"
    # Fallback for legacy calls without agent_id (not recommended for sheets)
    return f"docling_{company_id}_{source}"

def get_company_doc_table(company_id: str, source: str = "general", agent_id: Optional[str] = None):
    """Returns the collection name for the company's documents."""
    collection = company_doc_collection(company_id, source, agent_id)
    ensure_collection(collection)
    return collection

def get_company_sendable_table(company_id: str):
    """Returns the collection name for the company's sendable files."""
    collection = company_sendable_collection(company_id)
    ensure_collection(collection)
    return collection

def company_sendable_collection(company_id: str) -> str:
    return f"sendable_files_{company_id}"

def ensure_collection(name: str):
    """Create Qdrant collection if missing with HNSW index tuned for latency and sparse vectors.
    
    Also handles migration: if collection exists but lacks named vectors (dense/sparse),
    it will be deleted and recreated with the correct schema.
    """
    try:
        collection_info = qdrant.get_collection(name)
        # Check if collection has the correct named vector configuration
        vectors_config = collection_info.config.params.vectors
        
        # If vectors_config is a dict with 'dense' key, schema is correct
        if isinstance(vectors_config, dict) and "dense" in vectors_config:
            return
        
        # Otherwise, collection has old schema (unnamed vectors) - need to recreate
        # WARNING: This will delete all existing data in the collection!
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Collection '{name}' has old schema without named vectors. Recreating with correct schema...")
        logger.warning(f"Current vectors_config type: {type(vectors_config)}, value: {vectors_config}")
        qdrant.delete_collection(name)
    except Exception:
        pass

    qdrant.recreate_collection(
        collection_name=name,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=EMBED_DIMS,
                distance=qmodels.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(
                index=qmodels.SparseIndexParams(
                    on_disk=False,
                )
            )
        },
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
    """Deterministic numeric ID for Qdrant (guaranteed within signed 64-bit range)."""
    INT64_MAX = (1 << 63) - 1
    h = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    v = int.from_bytes(h[:8], byteorder="big", signed=False)  # 64-bit unsigned
    return v & INT64_MAX  # Constrain to signed 64-bit range (0 to 2^63-1)

def embed_texts(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """
    Generate embeddings using Google Gemini embedding model.
    
    Args:
        texts: List of texts to embed
        task_type: Task type for embeddings:
            - "RETRIEVAL_DOCUMENT" for indexing documents
            - "RETRIEVAL_QUERY" for search queries
            - "SEMANTIC_SIMILARITY" for similarity comparisons
    
    Returns:
        List of embedding vectors
    """
    try:
        # Gemini supports batching - embed all texts at once
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=texts,
            task_type=task_type,
            output_dimensionality=EMBED_DIMS,
        )
        # result.embedding is a list of embeddings when content is a list
        return result['embedding']
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Google API rate limit exceeded. Please try again in a few moments."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embeddings: {error_msg}"
            )

def embed_text_single(text: str, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
    """
    Generate embedding for a single text using Gemini.
    Optimized for query embeddings with appropriate task type.
    """
    try:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type=task_type,
            output_dimensionality=EMBED_DIMS,
        )
        return result['embedding']
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Google API rate limit exceeded. Please try again."
            )
        raise HTTPException(status_code=500, detail=f"Embedding failed: {error_msg}")

def stable_hash(text: str) -> int:
    """Create a stable 32-bit unsigned hash from text (suitable for Qdrant sparse indices)."""
    # Qdrant sparse vector indices are uint32; keep within [0, 2^32-1]
    UINT32_MAX = (1 << 32) - 1
    hash_bytes = hashlib.sha1(text.encode('utf-8')).digest()
    # Use first 4 bytes for 32-bit range
    hash_int = int.from_bytes(hash_bytes[:4], byteorder='big', signed=False)
    return hash_int & UINT32_MAX

def create_sparse_vector(text: str, collection_name: Optional[str] = None, update_stats: bool = False) -> qmodels.SparseVector:
    """
    Create BM25-style sparse vector from text using Unicode-aware tokenization.
    Supports all languages including Cyrillic, CJK, Arabic, etc.
    
    Args:
        text: Input text to vectorize
        collection_name: Optional collection name for BM25 stats tracking
        update_stats: If True and collection_name provided, updates BM25 stats in Redis
    
    Returns:
        Qdrant SparseVector with term indices and frequencies
    """
    # \p{L} = any Unicode letter (Latin, Cyrillic, CJK, Arabic, etc.)
    # \p{N} = any Unicode number (digits in any script)
    tokens = regex.findall(r'[\p{L}\p{N}]+', text.lower())
    
    # Filter out very short tokens (optional, but recommended)
    tokens = [t for t in tokens if len(t) >= 2 or t.isdigit()]
    
    # Update BM25 statistics if requested
    if update_stats and collection_name:
        bm25_update_stats_redis(collection_name, tokens)
    
    # Count term frequencies
    term_freq: Dict[str, int] = {}
    for token in tokens:
        term_freq[token] = term_freq.get(token, 0) + 1
    
    # Convert to sparse vector format (index: stable hash of term, value: frequency)
    indices = []
    values = []
    for term, freq in term_freq.items():
        # Use stable hash to create consistent numeric indices across processes
        index = stable_hash(term)
        indices.append(index)
        values.append(float(freq))
    
    return qmodels.SparseVector(indices=indices, values=values)

def create_sparse_vector_query_bm25(collection_name: str, text: str) -> qmodels.SparseVector:
    """
    Create BM25 sparse vector for query search (IDF-weighted term frequencies).
    Uses cached BM25 stats to reduce Redis round-trips.
    
    Query BM25 formula (simplified):
    weight(term) = tf(term) * IDF(term)
    
    Args:
        collection_name: Collection name for BM25 statistics
        text: Query text to vectorize
    
    Returns:
        Qdrant SparseVector with IDF-weighted term frequencies
    """
    tokens = _tokenize_multilingual(text)
    # Use cached stats for queries (don't need fresh stats every time)
    s = _get_stats_cached(collection_name)
    tf = Counter(tokens)
    
    idx, val = [], []
    for term, f in tf.items():
        idf = _idf(term, s["N"], s["DF"])
        w = float(f) * float(idf)
        idx.append(stable_hash(term))
        val.append(w)
    
    return qmodels.SparseVector(indices=idx, values=val)

def create_sparse_vector_doc_tf_norm(
    collection_name: str,
    text: str,
    k1: float = 1.2,
    b: float = 0.75,
) -> qmodels.SparseVector:
    tokens = _tokenize_multilingual(text)
    tf = Counter(tokens)
    dl = sum(tf.values())

    # Read-only snapshot of avgdl (cached is fine)
    s = _get_stats_cached(collection_name)
    avgdl = max(1.0, float(s.get("AVGDL", 200.0)))

    idx, val = [], []
    norm = (1.0 - b + b * (dl / avgdl))
    for term, f in tf.items():
        denom = f + k1 * norm
        w = (f * (k1 + 1.0)) / max(1e-9, denom)  # NO IDF in doc
        idx.append(stable_hash(term))
        val.append(float(w))

    return qmodels.SparseVector(indices=idx, values=val), tokens

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[Dict[str, Any]], 
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combine dense and sparse search results using Reciprocal Rank Fusion (RRF).
    
    Args:
        dense_results: Results from dense vector search (must include 'id' field)
        sparse_results: Results from sparse vector search (must include 'id' field)
        k: RRF constant (default: 60)
    
    Returns:
        Fused and re-ranked results
    """
    # Create a map of document ID to combined score
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}
    
    # Process dense results
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = str(doc["id"])  # Use stable Qdrant point ID as string
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_map.setdefault(doc_id, doc)
    
    # Process sparse results
    for rank, doc in enumerate(sparse_results, start=1):
        doc_id = str(doc["id"])  # Use stable Qdrant point ID as string
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_map.setdefault(doc_id, doc)
    
    # Sort by combined RRF score and return with scores
    return [
        {**doc_map[doc_id], "rrf_score": score}
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]

def _weighted_rrf(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[Dict[str, Any]], 
    k: int = 60,
    w_dense: float = 1.0,
    w_sparse: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Weighted Reciprocal Rank Fusion for dynamic weight adjustment.
    Useful for boosting sparse results on short queries.
    
    Args:
        dense_results: Results from dense vector search
        sparse_results: Results from sparse vector search
        k: RRF constant (default: 60)
        w_dense: Weight for dense results (default: 1.0)
        w_sparse: Weight for sparse results (default: 1.0)
    
    Returns:
        Fused and re-ranked results with rrf_score
    """
    from collections import defaultdict
    scores = defaultdict(float)
    doc = {}
    
    # Process dense results with weight
    for r, d in enumerate(dense_results, 1):
        did = str(d["id"])
        scores[did] += w_dense / (k + r)
        doc.setdefault(did, d)
    
    # Process sparse results with weight
    for r, s in enumerate(sparse_results, 1):
        did = str(s["id"])
        scores[did] += w_sparse / (k + r)
        doc.setdefault(did, s)
    
    # Create fused results
    fused = [{**doc[did], "rrf_score": sc} for did, sc in scores.items()]
    return sorted(fused, key=lambda x: x["rrf_score"], reverse=True)

# ------------------------------------------------------------------------------
# Small in-memory LRU cache for query embeddings (speeds up repeated searches)
# ------------------------------------------------------------------------------
_EMB_CACHE_MAX = int(os.getenv("QUERY_EMBED_CACHE_SIZE", "500"))
_emb_cache: "OrderedDict[str, List[float]]" = OrderedDict()

def get_cached_query_embedding(query: str) -> List[float]:
    """
    Synchronous Gemini embedding with LRU cache.
    Uses RETRIEVAL_QUERY task type for optimal search performance.
    """
    key = query.strip()
    if not key:
        return []
    vec = _emb_cache.get(key)
    if vec is not None:
        _emb_cache.move_to_end(key)
        return vec
    # Compute embedding with Gemini
    vec = embed_text_single(key, task_type="RETRIEVAL_QUERY")
    _emb_cache[key] = vec
    if len(_emb_cache) > _EMB_CACHE_MAX:
        _emb_cache.popitem(last=False)
    return vec

async def get_cached_query_embedding_async(query: str) -> List[float]:
    """
    Async Gemini embedding with cache - runs in thread pool to not block event loop.
    Gemini SDK is sync, so we wrap it with asyncio.to_thread for async context.
    """
    key = query.strip()
    if not key:
        return []
    vec = _emb_cache.get(key)
    if vec is not None:
        _emb_cache.move_to_end(key)
        return vec
    # Run sync Gemini call in thread pool
    vec = await asyncio.to_thread(embed_text_single, key, "RETRIEVAL_QUERY")
    _emb_cache[key] = vec
    if len(_emb_cache) > _EMB_CACHE_MAX:
        _emb_cache.popitem(last=False)
    return vec

def qdrant_sparse_search(table: str, query: str, limit: int) -> List[Dict[str, Any]]:
    """
    Sparse keyword-style search using Qdrant sparse vectors with BM25.
    Returns formatted results compatible with dense results for RRF.
    """
    sparse_vec = create_sparse_vector_query_bm25(table, query)
    hits = qdrant.search(
        collection_name=table,
        query_vector=("sparse", sparse_vec),
        limit=limit,
        with_payload=True,
    )
    results: List[Dict[str, Any]] = []
    for hit in hits:
        pl = hit.payload or {}
        md = pl.get("metadata", {})
        pn_raw = md.get("page_numbers")
        page_numbers = None
        if isinstance(pn_raw, str):
            try:
                page_numbers = json.loads(pn_raw)
            except Exception:
                page_numbers = None
        elif isinstance(pn_raw, list):
            page_numbers = pn_raw

        results.append({
            "id":           str(hit.id),
            "text":         pl.get("text"),
            "filename":     md.get("filename"),
            "page_numbers": page_numbers,
            "title":        md.get("title"),
            "url":          md.get("url"),
            "score":        float(hit.score) if hit.score is not None else 0.0,
        })
    return results


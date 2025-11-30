# routers/documents.py
import os
import json
import time
import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from docling.chunking import HybridChunker

from qdrant_client.http import models as qmodels

# Импортируем все необходимое

from config import (
    client, bucket, qdrant, tokenizer, MAX_TOKENS,
    RAG_USE_LLM_UNDERSTAND_DEFAULT, RAG_NATURAL_CHUNKING_DEFAULT,
    RAG_SOFT_MAX_TOKENS, RAG_HARD_MAX_TOKENS,
    OPENAI_RERANKER_ENABLED, JINA_RERANKER_ENABLED, JINA_API_KEY, OPENAI_RERANKER_TOP_K,
    OPENAI_RERANKER_MODEL, OPENAI_RERANKER_TIMEOUT, 
    JINA_RERANKER_MODEL, JINA_RERANKER_ENDPOINT, JINA_RERANKER_TIMEOUT,
    converter, chunker
)
from utils.processing import (
    safe_decode_filename, upload_to_gcs, 
    _extract_sections_from_markdown, _build_natural_chunks, _llm_understand_sections, _sse_format
)
from utils.qdrant_helpers import (
    get_company_doc_table, sha_id, embed_texts,
    get_cached_query_embedding, get_cached_query_embedding_async,
    _weighted_rrf, _tokenize_multilingual, _emb_cache,
    create_sparse_vector_doc_bm25, create_sparse_vector_query_bm25
)
from utils.bm25_helpers import bm25_decrement_stats, _tokenize_multilingual as bm25_tokenize

from utils.reranker import rerank_with_openai, rerank_with_jina # и т.д.

# Настраиваем логирование
logger = logging.getLogger(__name__)

# Создаем роутер
# prefix - это общий путь для всех эндпоинтов в этом файле
# tags - для группировки в документации Swagger/OpenAPI
router = APIRouter(
    prefix="/companies/{companyId}",
    tags=["Documents"]
)


@router.post("/process-document")
async def process_document(
    companyId: str,
    file: UploadFile = File(...),
    llmEnrich: Optional[bool] = Query(None, description="Use LLM understanding pass"),
    naturalChunking: Optional[bool] = Query(None, description="Use paragraph/sentence natural chunking"),
    softMaxTokens: Optional[int] = Query(None, description="Soft token cap for natural chunks"),
    hardMaxTokens: Optional[int] = Query(None, description="Hard token cap for natural chunks"),
    stream: Optional[bool] = Query(False, description="Stream processing updates via SSE"),
    source: str = Query("general", description="Document source: 'general' for uploaded docs, 'sheets' for Google Sheets/Docs"),
    agentId: Optional[str] = Query(None, description="Agent ID for source-specific documents (e.g., Google Sheets)"),
    allowUpdate: bool = Query(False, description="Allow updating existing document (deletes old chunks first, for resync)"),
):
    safe_filename = safe_decode_filename(file.filename)

    # 1) Check for duplicates
    # For source-specific collections (e.g., "sheets"), use agent_id for isolation
    table = get_company_doc_table(companyId, source, agentId)
    filter_q = qmodels.Filter(
        must=[qmodels.FieldCondition(key="metadata.filename", match=qmodels.MatchValue(value=safe_filename))]
    )
    dup_count = qdrant.count(collection_name=table, count_filter=filter_q, exact=True).count
    if dup_count > 0:
        if allowUpdate:
            # Delete existing chunks to allow re-sync (used by Google Docs webhook updates)
            logger.info(f"allowUpdate=True: Deleting {dup_count} existing chunks for '{safe_filename}' before re-indexing")

            # Retrieve ALL existing chunks to decrement BM25 stats before deletion
            # Must paginate through all results, not just first 1000
            try:
                total_decremented = 0
                cursor = None
                while True:
                    existing_points, cursor = qdrant.scroll(
                        collection_name=table,
                        scroll_filter=filter_q,
                        limit=1000,
                        offset=cursor,
                        with_payload=True,
                    )
                    if not existing_points:
                        break

                    # Collect all tokens from existing chunks and decrement BM25 stats
                    for point in existing_points:
                        if point.payload and point.payload.get("text"):
                            old_text = point.payload["text"]
                            old_title = (point.payload.get("metadata") or {}).get("title", "")
                            sparse_source = f"{old_title}. {old_text}" if old_title else old_text
                            old_tokens = bm25_tokenize(sparse_source)
                            bm25_decrement_stats(table, old_tokens)
                            total_decremented += 1

                    if cursor is None:
                        break

                logger.info(f"Decremented BM25 stats for {total_decremented} chunks")
            except Exception as bm25_err:
                logger.warning(f"Failed to decrement BM25 stats: {bm25_err}")

            qdrant.delete(collection_name=table, points_selector=qmodels.FilterSelector(filter=filter_q))
        else:
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
                soft_cap = int(softMaxTokens) if softMaxTokens else RAG_SOFT_MAX_TOKENS
                hard_cap = int(hardMaxTokens) if hardMaxTokens else RAG_HARD_MAX_TOKENS

                # Use markdown directly from Docling (no LLM restructuring)
                final_markdown = markdown_output

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
                        logger.info(f"Indexing chunk {i}: title='{item['metadata']['title']}', text_preview='{item['text'][:100]}...'")
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
                        logger.info(f"Indexing chunk {i}: title='{item['metadata']['title']}', text_preview='{item['text'][:100]}...'")
                        yield _sse_format("chunk", {"index": i, "title": item["metadata"]["title"], "preview": item["text"][:160], "pages": page_nums})

                if not to_store:
                    raise ValueError("No chunks were prepared for storage")

                # Store in Qdrant
                yield _sse_format("status", {"step": "store", "message": f"Storing {len(to_store)} chunks"})
                
                # Convert to Qdrant format with dense and sparse vectors
                # Include section title in the text we embed so titles influence search
                embed_inputs = []
                for item in to_store:
                    title = (item.get("metadata") or {}).get("title")
                    if title:
                        embed_inputs.append(f"Title: {title}\n\n{item['text']}")
                    else:
                        embed_inputs.append(item["text"])
                
                # Generate dense vectors
                dense_vectors = embed_texts(embed_inputs)
                
                # Build points with both dense and sparse vectors (Qdrant-only)
                points = []
                for i, (item, dense_vec) in enumerate(zip(to_store, dense_vectors)):
                    point_id = sha_id(companyId, safe_filename, str(i))
                    
                    payload = {
                        "text": item["text"],
                        "metadata": item["metadata"]
                    }
                    
                    # Build point with named dense and sparse vectors
                    title = (item.get("metadata") or {}).get("title")
                    sparse_source = f"{title}. {item['text']}" if title else item["text"]
                    sparse_vec = create_sparse_vector_doc_bm25(table, sparse_source)
                    # Log tokens being indexed
                    tokens = _tokenize_multilingual(sparse_source)
                    logger.info(f"Tokens for chunk {i}: {tokens[:20]}{'...' if len(tokens) > 20 else ''} (total {len(tokens)} tokens)")
                    # IMPORTANT: Pass both dense and sparse vectors in the vector parameter
                    point = qmodels.PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense_vec,
                            "sparse": qmodels.SparseVector(
                                indices=sparse_vec.indices,
                                values=sparse_vec.values
                            )
                        },
                        payload=payload
                    )
                    points.append(point)
                
                qdrant.upsert(collection_name=table, wait=True, points=points)
                total_count = qdrant.count(collection_name=table, exact=False).count
                
                yield _sse_format("done", {
                    "message": "Document processed and embeddings stored successfully.",
                    "row_count": int(total_count),
                    "url": gcs_url,
                    "natural": bool(use_natural),
                    "llm": bool(use_llm),
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
        soft_cap = int(softMaxTokens) if softMaxTokens else RAG_SOFT_MAX_TOKENS
        hard_cap = int(hardMaxTokens) if hardMaxTokens else RAG_HARD_MAX_TOKENS

        # Use markdown directly from Docling (no LLM restructuring)
        final_markdown = markdown_output

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
                logger.info(f"Indexing chunk {i}: title='{item['metadata']['title']}', text_preview='{item['text'][:100]}...'")
                try:
                    print({"chunk_index": i, "title": item["metadata"]["title"], "preview": item["text"][:160]})
                except Exception:
                    pass  # Ignore print errors
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
                logger.info(f"Indexing chunk {i}: title='{item['metadata']['title']}', text_preview='{item['text'][:100]}...'")
                try:
                    print({"chunk_index": i, "title": item["metadata"]["title"], "preview": item["text"][:160], "pages": page_nums})
                except Exception:
                    pass  # Ignore print errors

        if not to_store:
            raise ValueError("No chunks were prepared for storage")

        # Convert to Qdrant format and store with dense and sparse vectors
        # Include section title in the text we embed so titles influence search
        embed_inputs = []
        for item in to_store:
            title = (item.get("metadata") or {}).get("title")
            if title:
                embed_inputs.append(f"Title: {title}\n\n{item['text']}")
            else:
                embed_inputs.append(item["text"])
        
        # Generate dense vectors
        dense_vectors = embed_texts(embed_inputs)
        
        # Build points with both dense and sparse vectors (Qdrant-only)
        points = []
        for i, (item, dense_vec) in enumerate(zip(to_store, dense_vectors)):
            point_id = sha_id(companyId, safe_filename, str(i))
            
            payload = {
                "text": item["text"],
                "metadata": item["metadata"]
            }
            
            # Build point with named dense and sparse vectors
            title = (item.get("metadata") or {}).get("title")
            sparse_source = f"{title}. {item['text']}" if title else item["text"]
            sparse_vec = create_sparse_vector_doc_bm25(table, sparse_source)
            # Log tokens being indexed
            tokens = _tokenize_multilingual(sparse_source)
            logger.info(f"Tokens for chunk {i}: {tokens[:20]}{'...' if len(tokens) > 20 else ''} (total {len(tokens)} tokens)")
            # IMPORTANT: Pass both dense and sparse vectors in the vector parameter
            point = qmodels.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": qmodels.SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values
                    )
                },
                payload=payload
            )
            points.append(point)
        
        qdrant.upsert(collection_name=table, wait=True, points=points)
        total_count = qdrant.count(collection_name=table, exact=False).count
        
        response_obj = {
            "message": "Document processed and embeddings stored successfully.",
            "row_count": int(total_count),
            "url": gcs_url,
            "natural": bool(use_natural),
            "llm": bool(use_llm),
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


@router.get("/search")
async def search_documents(
    companyId: str,
    query: str = Query(...),
    limit: int = Query(5),
    useReranker: bool = Query(True, description="Use reranker for better results"),
    rerankTopK: Optional[int] = Query(None, description="Number of docs to retrieve before reranking"),
    rerankerProvider: Optional[str] = Query(None, description="'openai' or 'jina' (overrides env flags)"),
    useHybrid: bool = Query(True, description="Use hybrid search (dense + sparse with RRF)"),
    source: str = Query("general", description="Document source: 'general' for uploaded docs, 'sheets' for Google Sheets/Docs"),
    agentId: Optional[str] = Query(None, description="Agent ID for source-specific documents (e.g., Google Sheets)"),
):
    """
    Search through company documents using hybrid semantic + keyword search with RRF fusion.
    Returns matching chunks with their metadata and relevance scores.
    Optimized for <500ms response time.
    """
    search_start = time.perf_counter()
    table = get_company_doc_table(companyId, source, agentId)
    
    # 1) Determine reranker provider and search limits FIRST (no I/O)
    provider = (rerankerProvider or "").strip().lower() if rerankerProvider else None
    use_openai = False
    use_jina = False
    if useReranker:
        if provider == "openai":
            use_openai = OPENAI_RERANKER_ENABLED
        elif provider == "jina":
            use_jina = JINA_RERANKER_ENABLED and bool(JINA_API_KEY)
        else:
            if OPENAI_RERANKER_ENABLED:
                use_openai = True
            elif JINA_RERANKER_ENABLED and bool(JINA_API_KEY):
                use_jina = True

    rerank_enabled = use_openai or use_jina
    search_limit = rerankTopK or (OPENAI_RERANKER_TOP_K if rerank_enabled else limit)
    
    # 2) Start embedding generation async while preparing sparse vector
    try:
        t0 = time.perf_counter()
        # Check cache first (instant if cached)
        if query.strip() in _emb_cache:
            query_dense_vec = _emb_cache[query.strip()]
            _emb_cache.move_to_end(query.strip())
            logger.debug(f"Query embedding cache hit in {(time.perf_counter()-t0)*1000:.1f}ms")
        else:
            # Use async embedding - doesn't block event loop
            query_dense_vec = await get_cached_query_embedding_async(query)
            logger.debug(f"Query embedding generated in {(time.perf_counter()-t0)*1000:.1f}ms")
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate_limit" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="OpenAI API rate limit exceeded. Please try again in a few moments."
            )
        raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {error_msg}")

    # 3) Hybrid search: dense + sparse with RRF fusion
    initial_results = []
    
    if useHybrid:
        try:
            # Create sparse vector (uses cached BM25 stats)
            query_sparse_vec = create_sparse_vector_query_bm25(table, query)
            
            t0 = time.perf_counter()
            # Run dense and sparse searches in parallel
            async def _dense_job():
                return await asyncio.to_thread(
                    qdrant.query_points,
                    collection_name=table,
                    query=query_dense_vec,
                    using="dense",
                    limit=search_limit,
                    with_payload=True,
                    score_threshold=None,
                )
            async def _sparse_job():
                return await asyncio.to_thread(
                    qdrant.query_points,
                    collection_name=table,
                    query=query_sparse_vec,
                    using="sparse",
                    limit=search_limit,
                    with_payload=True,
                )
            dense_result, sparse_result = await asyncio.gather(_dense_job(), _sparse_job())
            dense_hits = dense_result.points
            sparse_hits = sparse_result.points
            logger.debug(f"Qdrant search: Dense={len(dense_hits)} Sparse={len(sparse_hits)} in {(time.perf_counter()-t0)*1000:.1f}ms")
            
            # Format results (optimized - single pass)
            def format_hits(hits):
                results = []
                for hit in hits:
                    pl = hit.payload or {}
                    md = pl.get("metadata", {})
                    pn_raw = md.get("page_numbers")
                    page_numbers = None
                    if isinstance(pn_raw, str):
                        try:
                            page_numbers = json.loads(pn_raw)
                        except Exception:
                            pass
                    elif isinstance(pn_raw, list):
                        page_numbers = pn_raw

                    results.append({
                        "id": str(hit.id),
                        "text": pl.get("text"),
                        "filename": md.get("filename"),
                        "page_numbers": page_numbers,
                        "title": md.get("title"),
                        "url": md.get("url"),
                        "score": float(hit.score) if hit.score is not None else 0.0
                    })
                return results
            
            dense_results = format_hits(dense_hits)
            sparse_results = format_hits(sparse_hits)
            
            # Apply weighted RRF fusion (boost sparse for short queries)
            q_toks = _tokenize_multilingual(query)
            if len(q_toks) <= 3:
                initial_results = _weighted_rrf(dense_results, sparse_results, w_dense=0.8, w_sparse=1.2)
            else:
                initial_results = _weighted_rrf(dense_results, sparse_results, w_dense=1.0, w_sparse=1.0)
            
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to dense-only: {e}")
            useHybrid = False
    
    # Fallback: dense-only search
    if not useHybrid:
        try:
            t0 = time.perf_counter()
            hits = await asyncio.to_thread(
                qdrant.query_points,
                collection_name=table,
                query=query_dense_vec,
                using="dense",
                limit=search_limit,
                with_payload=True,
                score_threshold=None,
            )
            hits = hits.points
            logger.debug(f"Dense-only search: {len(hits)} results in {(time.perf_counter()-t0)*1000:.1f}ms")
            
            for hit in hits:
                pl = hit.payload or {}
                md = pl.get("metadata", {})
                pn_raw = md.get("page_numbers")
                page_numbers = None
                if isinstance(pn_raw, str):
                    try:
                        page_numbers = json.loads(pn_raw)
                    except Exception:
                        pass
                elif isinstance(pn_raw, list):
                    page_numbers = pn_raw

                initial_results.append({
                    "id": str(hit.id),
                    "text": pl.get("text"),
                    "filename": md.get("filename"),
                    "page_numbers": page_numbers,
                    "title": md.get("title"),
                    "url": md.get("url"),
                    "score": float(hit.score) if hit.score is not None else 0.0
                })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search error: {e}")
    
    # 4) Apply reranking if enabled
    final_results = initial_results
    rerank_metadata = {"reranked": False, "hybrid": useHybrid}

    if rerank_enabled and len(initial_results) > 1:
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
            logger.warning(f"Reranking failed: {rerank_error}")
            final_results = initial_results[:limit]
    else:
        final_results = initial_results[:limit]
    
    total_time_ms = int((time.perf_counter() - search_start) * 1000)
    logger.info(f"Search completed in {total_time_ms}ms for query: '{query[:50]}...'")
    
    return {
        "results": final_results,
        "search_time_ms": total_time_ms,
        **rerank_metadata
    }

@router.delete("/delete-document")
async def delete_document(
    companyId: str,
    filename: str = Query(...),
    source: str = Query("general", description="Document source: 'general' for uploaded docs, 'sheets' for Google Sheets/Docs"),
    agentId: Optional[str] = Query(None, description="Agent ID for source-specific documents (e.g., Google Sheets)"),
):
    """
    Delete a document and its chunks from both Qdrant and GCS.
    The filename parameter should be just the filename, not a JSON object.
    """
    # 1) Get the table
    table = get_company_doc_table(companyId, source, agentId)

    # 2) Count matching records before deletion
    filter_q = qmodels.Filter(
        must=[qmodels.FieldCondition(key="metadata.filename", match=qmodels.MatchValue(value=filename))]
    )
    
    # Get count for response
    count_result = qdrant.count(collection_name=table, count_filter=filter_q, exact=True)
    deleted = count_result.count

    # 3) Delete from Qdrant
    qdrant.delete(collection_name=table, points_selector=qmodels.FilterSelector(filter=filter_q))

    # Meilisearch removal: no secondary index to delete from

    # 4) Delete from GCS (only for general uploaded documents, not for sheets/docs)
    # Google Sheets/Docs are streamed directly as markdown and don't have GCS files
    if source == "general":
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
@router.get("/download-document")
async def download_document(companyId: str, filename: str = Query(...)):
    path = f"uploads/{companyId}/documents/{filename}"
    blob = bucket.blob(path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="File not found in GCS")

    signed_url = blob.generate_signed_url(expiration=15 * 60)  # 15 минут
    return JSONResponse({"url": signed_url})


@router.post("/transcribe-document")
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

@router.get("/documents/list")
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


@router.get("/documents/content")
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
                "page_numbers": metadata.get("page_numbers"),
                "title": metadata.get("title")  # Store section title with each chunk
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
        
        # Collect all section titles from chunks (skip None/empty titles)
        section_titles = []
        seen_titles = set()
        for chunk in doc["chunks"]:
            # Get the title from the chunk's metadata if stored there
            chunk_title = None
            if isinstance(chunk, dict):
                chunk_title = chunk.get("title")
            
            if chunk_title and chunk_title not in seen_titles:
                section_titles.append(chunk_title)
                seen_titles.add(chunk_title)
        
        result.append({
            "filename": doc["filename"],
            "title": doc["title"],  # First encountered section title (kept for compatibility)
            "section_titles": section_titles,  # All unique section titles in order
            "url": doc["url"],
            "content": full_text
        })
    try:
        print(f"Returning {len(result)} documents with total content length: {sum(len(d.get('content', '')) for d in result)}")
    except Exception:
        pass  # Ignore print errors
    return {"documents": result}

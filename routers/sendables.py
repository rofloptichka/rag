# routers/sendables.py
import os
import logging
import traceback
from datetime import datetime
from qdrant_client.http import models as qmodels
import json
import time
import asyncio
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Form
from fastapi.responses import StreamingResponse
import cloudinary
import cloudinary.uploader


# Импортируем все необходимое

from schemas import UpdateSendableDescription

from config import (
    client, bucket, qdrant,
    RAG_USE_LLM_UNDERSTAND_DEFAULT, RAG_NATURAL_CHUNKING_DEFAULT,
    RAG_SOFT_MAX_TOKENS, RAG_HARD_MAX_TOKENS
)
from utils.processing import (
    safe_decode_filename, upload_to_gcs, 
)
from utils.qdrant_helpers import (
    sha_id, embed_texts, get_company_sendable_table, get_cached_query_embedding,
    create_sparse_vector_query_bm25, _weighted_rrf, _tokenize_multilingual,
    create_sparse_vector_doc_bm25
)

from utils.reranker import rerank_with_openai, rerank_with_jina # и т.д.

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logger = logging.getLogger("sendables")
logger.setLevel(logging.DEBUG)

# Create console handler with formatting
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def _debug_request_info(endpoint: str, companyId: str, **kwargs):
    """Log request information for debugging"""
    logger.info("=" * 70)
    logger.info(f"🚀 ENDPOINT: {endpoint}")
    logger.info(f"📋 Company ID: {companyId}")
    for key, value in kwargs.items():
        if key == "content" and value:
            logger.debug(f"   {key}: <{len(value)} bytes>")
        elif key == "description" and value and len(str(value)) > 100:
            logger.debug(f"   {key}: {str(value)[:100]}...")
        else:
            logger.debug(f"   {key}: {value}")
    logger.info("-" * 70)


def _debug_timing(operation: str, start_time: float):
    """Log timing information"""
    elapsed = time.time() - start_time
    logger.debug(f"⏱️  {operation}: {elapsed:.3f}s")
    return time.time()


# Создаем роутер
# prefix - это общий путь для всех эндпоинтов в этом файле
# tags - для группировки в документации Swagger/OpenAPI
router = APIRouter(
    prefix="/companies/{companyId}",
    tags=["Sendables"]
)


@router.post("/process-sendable-file")
async def process_sendable_file(
    companyId:   str,
    file:        UploadFile = File(...),
    description: str        = Form(...)
):
    total_start = time.time()
    step_time = total_start
    
    _debug_request_info(
        "POST /process-sendable-file",
        companyId,
        filename=file.filename,
        content_type=file.content_type,
        description_length=len(description) if description else 0
    )
    
    safe_name = safe_decode_filename(file.filename)
    logger.debug(f"📁 Safe filename: {safe_name}")
    
    table = get_company_sendable_table(companyId)
    logger.debug(f"📦 Qdrant collection: {table}")
    step_time = _debug_timing("Collection name resolution", step_time)

    # 1) Check for duplicates
    logger.debug("🔍 Checking for duplicate files...")
    filter_q = qmodels.Filter(
        must=[qmodels.FieldCondition(key="metadata.filename", match=qmodels.MatchValue(value=safe_name))]
    )
    try:
        dup_count = qdrant.count(collection_name=table, count_filter=filter_q, exact=True).count
        logger.debug(f"   Duplicate check result: {dup_count} existing record(s)")
        step_time = _debug_timing("Duplicate check", step_time)
    except Exception as e:
        logger.error(f"❌ Failed to check duplicates: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to check duplicates: {str(e)}")
    
    if dup_count > 0:
        logger.warning(f"⚠️  Duplicate found: '{safe_name}' already exists for company {companyId}")
        raise HTTPException(
            status_code=400,
            detail=f"Sendable '{safe_name}' already processed for company {companyId}"
        )

    ### NEW: Generate a unique, sequential index for the sendable ###
    logger.debug("🔢 Calculating next sequential index...")
    # 1.1) Calculate the next index by scanning existing indices
    indices = []
    cursor = None
    scroll_count = 0
    while True:
        batch, cursor = qdrant.scroll(
            collection_name=table, 
            with_payload=True, 
            limit=1000, 
            offset=cursor
        )
        scroll_count += 1
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
    
    logger.debug(f"   Scanned {scroll_count} batch(es), found {len(indices)} existing indices")
    
    # The new index is the highest existing index + 1
    max_index = max(indices) if indices else 0
    new_index_num = max_index + 1
    sendable_index = f"SF-{new_index_num}"
    logger.info(f"📌 Generated new index: {sendable_index} (max existing: {max_index})")
    step_time = _debug_timing("Index generation", step_time)
    ### END NEW ###


    # 2) Read content and upload to GCS
    logger.debug("📤 Reading file content...")
    content = await file.read()
    logger.debug(f"   File size: {len(content)} bytes ({len(content)/1024:.2f} KB)")
    step_time = _debug_timing("File read", step_time)
    
    logger.debug("☁️  Uploading to Google Cloud Storage...")
    try:
        gcs_url = upload_to_gcs(
            content,
            companyId,
            "sendables",
            safe_name,
            file.content_type or "application/octet-stream"
        )
        logger.info(f"   ✅ GCS upload successful: {gcs_url[:80]}...")
        step_time = _debug_timing("GCS upload", step_time)
    except Exception as e:
        logger.error(f"❌ GCS upload failed: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"GCS upload failed: {str(e)}")

    # 2.1) If the file is an image, upload to Cloudinary as well
    cloudinary_url = None
    cloudinary_public_id = None
    if file.content_type and file.content_type.startswith("image/"):
        logger.debug(f"🖼️  Image detected ({file.content_type}), uploading to Cloudinary...")
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
            logger.info(f"   ✅ Cloudinary upload successful: {cloudinary_url}")
            logger.debug(f"   Public ID: {cloudinary_public_id}")
            step_time = _debug_timing("Cloudinary upload", step_time)
        except Exception as e:
            logger.warning(f"⚠️  Cloudinary upload failed (non-fatal): {e}")
            logger.debug(traceback.format_exc())
    else:
        logger.debug(f"📄 Non-image file type: {file.content_type}, skipping Cloudinary")


    # 3) Generate embedding for the description using Gemini
    logger.debug(f"🧠 Generating embedding for description ({len(description)} chars)...")
    logger.debug(f"   Description preview: {description[:150]}..." if len(description) > 150 else f"   Description: {description}")
    try:
        dense_embeddings = embed_texts([description], task_type="RETRIEVAL_DOCUMENT")
        dense_embedding = dense_embeddings[0]
        logger.debug(f"   ✅ Embedding generated: {len(dense_embedding)} dimensions")
        step_time = _debug_timing("Embedding generation", step_time)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Embedding generation failed: {error_msg}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embedding: {error_msg}"
        )

    # 4) Prepare record and save to Qdrant with dense and sparse vectors
    logger.debug("📝 Preparing Qdrant record...")
    metadata = {
        "index": sendable_index,
        "file_type": (file.content_type or os.path.splitext(safe_name)[1]),
        "filename": safe_name,
        "url": gcs_url,
        "cloudinary_url": cloudinary_url,
        "cloudinary_public_id": cloudinary_public_id,
    }
    logger.debug(f"   Metadata: {json.dumps(metadata, indent=2, default=str)}")

    payload = {
        "text": description,
        "metadata": metadata
    }
    
    point_id = sha_id(companyId, "sendable", safe_name)
    logger.debug(f"   Point ID (SHA): {point_id}")
    
    # Build point with named dense and sparse vectors using BM25
    logger.debug("📊 Creating sparse BM25 vector...")
    sparse_vec = create_sparse_vector_doc_bm25(table, description)
    logger.debug(f"   Sparse vector: {len(sparse_vec.indices)} non-zero entries")
    step_time = _debug_timing("Sparse vector creation", step_time)
    
    # IMPORTANT: Pass both dense and sparse vectors in the vector parameter
    point = qmodels.PointStruct(
        id=point_id,
        vector={
            "dense": dense_embedding,
            "sparse": qmodels.SparseVector(
                indices=sparse_vec.indices,
                values=sparse_vec.values
            )
        },
        payload=payload
    )
    
    logger.debug("💾 Upserting to Qdrant...")
    try:
        qdrant.upsert(collection_name=table, points=[point], wait=True)
        logger.info(f"   ✅ Qdrant upsert successful")
        step_time = _debug_timing("Qdrant upsert", step_time)
    except Exception as e:
        logger.error(f"❌ Qdrant upsert failed: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to save to Qdrant: {str(e)}")

    ### MODIFIED ###
    total_count = qdrant.count(collection_name=table, exact=False).count
    total_elapsed = time.time() - total_start
    
    logger.info("=" * 70)
    logger.info(f"✅ SENDABLE PROCESSED SUCCESSFULLY")
    logger.info(f"   Index: {sendable_index}")
    logger.info(f"   Filename: {safe_name}")
    logger.info(f"   Total records in collection: {total_count}")
    logger.info(f"   Total processing time: {total_elapsed:.3f}s")
    logger.info("=" * 70)
    
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
@router.put("/update-sendable-description")
async def update_sendable_description(
    companyId: str,
    data: UpdateSendableDescription,
    filename: str = Query(...),
   
):
    """
    Updates the description for a sendable file.
    """
    total_start = time.time()
    step_time = total_start
    
    _debug_request_info(
        "PUT /update-sendable-description",
        companyId,
        filename=filename,
        new_description_length=len(data.new_description) if data.new_description else 0
    )
    
    table = get_company_sendable_table(companyId)
    safe_name = safe_decode_filename(filename)
    logger.debug(f"📦 Collection: {table}, Safe filename: {safe_name}")

    new_description = data.new_description

    # 1) Find the record to update using Qdrant
    logger.debug("🔍 Searching for existing record...")
    filter_q = qmodels.Filter(
        must=[qmodels.FieldCondition(key="metadata.filename", match=qmodels.MatchValue(value=safe_name))]
    )
    try:
        found_records, _ = qdrant.scroll(
            collection_name=table, 
            scroll_filter=filter_q, 
            with_payload=True, 
            with_vectors=True,
            limit=2
        )
        logger.debug(f"   Found {len(found_records)} record(s)")
        step_time = _debug_timing("Record lookup", step_time)
    except Exception as e:
        logger.error(f"❌ Failed to search records: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    if not found_records:
        logger.warning(f"⚠️  No record found for filename '{safe_name}'")
        raise HTTPException(
            status_code=404,
            detail=f"No sendable with filename '{safe_name}' found."
        )
    elif len(found_records) > 1:
        logger.error(f"❌ Data integrity issue: {len(found_records)} records found for '{safe_name}'")
        raise HTTPException(
            status_code=500,
            detail=f"Multiple sendables with filename '{safe_name}' found. Data integrity issue."
        )
    
    logger.debug(f"   Existing record ID: {found_records[0].id}")

    # 2) Recalculate embeddings for the new description using Gemini
    logger.debug(f"🧠 Generating new embedding ({len(new_description)} chars)...")
    try:
        new_dense_embeddings = embed_texts([new_description], task_type="RETRIEVAL_DOCUMENT")
        new_dense_embedding = new_dense_embeddings[0]
        logger.debug(f"   ✅ New embedding generated: {len(new_dense_embedding)} dimensions")
        step_time = _debug_timing("Embedding generation", step_time)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Embedding generation failed: {error_msg}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embedding: {error_msg}"
        )

    # 3) Update with new description and dense vectors only
    point = found_records[0]
    payload = point.payload or {}
    old_description = payload.get("text", "")
    payload["text"] = new_description
    
    logger.debug(f"📝 Updating record...")
    logger.debug(f"   Old description: {old_description[:100]}..." if len(old_description) > 100 else f"   Old description: {old_description}")
    logger.debug(f"   New description: {new_description[:100]}..." if len(new_description) > 100 else f"   New description: {new_description}")
    
    # Build point with named dense + sparse vectors using BM25
    logger.debug("📊 Creating new sparse BM25 vector...")
    new_sparse = create_sparse_vector_doc_bm25(table, new_description)
    logger.debug(f"   Sparse vector: {len(new_sparse.indices)} non-zero entries")
    step_time = _debug_timing("Sparse vector creation", step_time)
    
    updated_point = qmodels.PointStruct(
        id=point.id,
        vector={
            "dense": new_dense_embedding,
            "sparse": qmodels.SparseVector(
                indices=new_sparse.indices,
                values=new_sparse.values
            )
        },
        payload=payload
    )
    
    try:
        qdrant.upsert(collection_name=table, points=[updated_point], wait=True)
        logger.info(f"   ✅ Qdrant update successful")
        step_time = _debug_timing("Qdrant upsert", step_time)
    except Exception as e:
        logger.error(f"❌ Qdrant update failed: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to update in Qdrant: {str(e)}")

    total_elapsed = time.time() - total_start
    logger.info(f"✅ Description updated for '{safe_name}' in {total_elapsed:.3f}s")
    
    return {"message": f"Description for '{safe_name}' updated successfully."}

@router.get("/search-sendable")
async def search_sendable(
    companyId: str,
    query:     str   = Query(...),
    limit:     int   = Query(5),
    useHybrid: bool  = Query(True, description="Use hybrid search (dense + sparse with RRF)")
):
    total_start = time.time()
    step_time = total_start
    
    _debug_request_info(
        "GET /search-sendable",
        companyId,
        query=query,
        limit=limit,
        useHybrid=useHybrid
    )
    
    table = get_company_sendable_table(companyId)
    logger.debug(f"📦 Collection: {table}")

    # 1) Get dense embedding for the query using Gemini (RETRIEVAL_QUERY for better matching)
    logger.debug(f"🧠 Generating query embedding...")
    try:
        query_embeddings = embed_texts([query], task_type="RETRIEVAL_QUERY")
        query_dense_vec = query_embeddings[0]
        logger.debug(f"   ✅ Query embedding: {len(query_dense_vec)} dimensions")
        step_time = _debug_timing("Query embedding", step_time)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Query embedding failed: {error_msg}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate query embedding: {error_msg}"
        )

    # 2) Hybrid search with RRF fusion
    results = []
    
    if useHybrid:
        logger.debug("🔀 Executing hybrid search (dense + sparse with RRF)...")
        try:
            # Dense search
            logger.debug("   Dense search...")
            dense_hits = qdrant.query_points(
                collection_name=table,
                query=query_dense_vec,
                using="dense",
                limit=limit * 2,  # Get more for fusion
                with_payload=True,
            ).points
            logger.debug(f"   Dense hits: {len(dense_hits)}")
            step_time = _debug_timing("Dense search", step_time)
            
            # Qdrant sparse search with BM25 query vector
            logger.debug("   Sparse (BM25) search...")
            sparse_hits = qdrant.query_points(
                collection_name=table,
                query=create_sparse_vector_query_bm25(table, query),
                using="sparse",
                limit=limit * 2,
                with_payload=True,
            ).points
            logger.debug(f"   Sparse hits: {len(sparse_hits)}")
            step_time = _debug_timing("Sparse search", step_time)
            
            # Format dense results
            def format_sendable_hits(hits):
                formatted = []
                for hit in hits:
                    pl = hit.payload or {}
                    md = pl.get("metadata", {})
                    formatted.append({
                        "id":        str(hit.id),  # Include Qdrant point ID as string for RRF
                        "index":     md.get("index"),
                        "text":      pl.get("text"),
                        "filename":  md.get("filename"),
                        "file_type": md.get("file_type"),
                        "url":       md.get("url"),
                        "cloudinary_url": md.get("cloudinary_url"),
                        "score":     float(hit.score) if hit.score is not None else 0.0
                    })
                return formatted
            
            dense_results = format_sendable_hits(dense_hits)
            sparse_results = format_sendable_hits(sparse_hits)
            
            # Apply weighted RRF fusion with boost for short queries
            q_toks = _tokenize_multilingual(query)
            short = (len(q_toks) <= 3)
            logger.debug(f"   Query tokens: {len(q_toks)}, short query boost: {short}")
            
            if short:
                fused = _weighted_rrf(dense_results, sparse_results,
                                     w_dense=0.8, w_sparse=1.2)
                logger.debug(f"   RRF weights: dense=0.8, sparse=1.2 (short query)")
            else:
                fused = _weighted_rrf(dense_results, sparse_results,
                                     w_dense=1.0, w_sparse=1.0)
                logger.debug(f"   RRF weights: dense=1.0, sparse=1.0 (normal)")
            
            results = fused[:limit]
            logger.debug(f"   Fused results: {len(results)}")
            step_time = _debug_timing("RRF fusion", step_time)
            
        except Exception as e:
            logger.warning(f"⚠️  Hybrid search failed, falling back to dense-only: {e}")
            logger.debug(traceback.format_exc())
            useHybrid = False
    
    # Fallback: dense-only search
    if not useHybrid:
        logger.debug("🔍 Executing dense-only search...")
        hits = qdrant.query_points(
            collection_name=table,
            query=query_dense_vec,
            using="dense",
            limit=limit,
            with_payload=True,
        ).points
        logger.debug(f"   Dense-only hits: {len(hits)}")
        step_time = _debug_timing("Dense-only search", step_time)
        
        for hit in hits:
            pl = hit.payload or {}
            md = pl.get("metadata", {})
            results.append({
                "id":        str(hit.id),  # Include Qdrant point ID as string
                "index":     md.get("index"),
                "text":      pl.get("text"),
                "filename":  md.get("filename"),
                "file_type": md.get("file_type"),
                "url":       md.get("url"),
                "cloudinary_url": md.get("cloudinary_url"),
                "score":     float(hit.score) if hit.score is not None else 0.0
            })

    total_elapsed = time.time() - total_start
    logger.info(f"✅ Search completed: {len(results)} results in {total_elapsed:.3f}s (hybrid={useHybrid})")
    
    # Log top results
    for i, r in enumerate(results[:3]):
        logger.debug(f"   #{i+1}: {r.get('index')} - {r.get('filename')} (score: {r.get('score', 0):.4f})")

    return {"results": results, "hybrid": useHybrid}


@router.get("/search-sendable-index")
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
    start_time = time.time()
    
    _debug_request_info(
        "GET /search-sendable-index",
        companyId,
        query=query,
        limit=limit
    )
    
    table = get_company_sendable_table(companyId)
    logger.debug(f"📦 Collection: {table}")

    # 1) Build a filter query to find the exact index in the metadata.
    #    This is much faster than a vector search.
    filter_q = f"metadata.index = '{query}'"
    logger.debug(f"🔍 Filter: {filter_q}")

    # 2) Execute the search using Qdrant filter
    filter_q = qmodels.Filter(
        must=[qmodels.FieldCondition(key="metadata.index", match=qmodels.MatchValue(value=query))]
    )
    try:
        found_records, _ = qdrant.scroll(
            collection_name=table, 
            scroll_filter=filter_q, 
            with_payload=True, 
            limit=limit
        )
        logger.debug(f"   Found {len(found_records)} record(s)")
    except Exception as e:
        logger.error(f"❌ Index search failed: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # 3) Check if a record was found.
    if not found_records:
        logger.warning(f"⚠️  No sendable found with index '{query}'")
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

    elapsed = time.time() - start_time
    logger.info(f"✅ Index search completed in {elapsed:.3f}s: found '{result.get('filename')}'")
    
    return result
# ==============================================================================
#  MODIFIED: delete_sendable
# ==============================================================================
@router.delete("/delete-sendable")
async def delete_sendable(
    companyId: str,
    filename:  Optional[str] = Query(None),
    index:     Optional[str] = Query(None)
):
    total_start = time.time()
    step_time = total_start
    
    _debug_request_info(
        "DELETE /delete-sendable",
        companyId,
        filename=filename,
        index=index
    )
    
    ### NEW: Validate input ###
    # Ensure exactly one identifier is provided
    if not (filename or index) or (filename and index):
        logger.warning("⚠️  Invalid input: must provide exactly one of 'filename' or 'index'")
        raise HTTPException(
            status_code=400,
            detail="You must provide exactly one of 'filename' or 'index'."
        )

    table = get_company_sendable_table(companyId)
    logger.debug(f"📦 Collection: {table}")

    # 1) Determine filter query and user-facing identifier
    if filename:
        safe_name = safe_decode_filename(filename)
        filter_q = qmodels.Filter(
            must=[qmodels.FieldCondition(key="metadata.filename", match=qmodels.MatchValue(value=safe_name))]
        )
        identifier = f"filename '{safe_name}'"
        logger.debug(f"🔍 Deleting by filename: {safe_name}")
    else:  # We know index is not None here because of the validation above
        filter_q = qmodels.Filter(
            must=[qmodels.FieldCondition(key="metadata.index", match=qmodels.MatchValue(value=index))]
        )
        identifier = f"index '{index}'"
        logger.debug(f"🔍 Deleting by index: {index}")

    # 2) Find the record(s) to delete *before* deleting from the DB
    # We need the metadata to clean up GCS and Cloudinary
    logger.debug("🔍 Searching for records to delete...")
    try:
        records_to_delete, _ = qdrant.scroll(
            collection_name=table, 
            scroll_filter=filter_q, 
            with_payload=True, 
            limit=10000
        )
        step_time = _debug_timing("Record lookup", step_time)
    except Exception as e:
        logger.error(f"❌ Failed to find records: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    deleted_count = len(records_to_delete)
    logger.debug(f"   Found {deleted_count} record(s) to delete")
    
    if deleted_count == 0:
        logger.warning(f"⚠️  No sendable found with {identifier}")
        raise HTTPException(
            status_code=404,
            detail=f"No sendable found with {identifier} for deletion."
        )

    # 3) Delete associated cloud assets (GCS and Cloudinary)
    logger.debug("☁️  Cleaning up cloud assets...")
    for i, point in enumerate(records_to_delete):
        payload = point.payload or {}
        metadata = payload.get("metadata", {})
        record_filename = metadata.get("filename", "unknown")
        logger.debug(f"   [{i+1}/{deleted_count}] Processing: {record_filename}")
        
        # 3.1) Delete from Cloudinary if applicable
        public_id = metadata.get("cloudinary_public_id")
        if public_id:
            try:
                cloudinary.uploader.destroy(public_id, resource_type="image")
                logger.info(f"      ✅ Deleted from Cloudinary: {public_id}")
            except Exception as e:
                logger.warning(f"      ⚠️  Failed to delete from Cloudinary: {e}")

        # 3.2) Delete blob from GCS
        gcs_filename = metadata.get("filename")
        if gcs_filename:
            path = f"uploads/{companyId}/sendables/{gcs_filename}"
            try:
                bucket.blob(path).delete()
                logger.info(f"      ✅ Deleted from GCS: {path}")
            except Exception as e:
                logger.warning(f"      ⚠️  Failed to delete from GCS: {e}")

    step_time = _debug_timing("Cloud asset cleanup", step_time)

    # 4) Delete from Qdrant (now that cloud assets are gone)
    logger.debug("💾 Deleting from Qdrant...")
    try:
        qdrant.delete(collection_name=table, points_selector=qmodels.FilterSelector(filter=filter_q))
        logger.info(f"   ✅ Deleted {deleted_count} record(s) from Qdrant")
        step_time = _debug_timing("Qdrant delete", step_time)
    except Exception as e:
        logger.error(f"❌ Qdrant delete failed: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to delete from Qdrant: {str(e)}")

    # Meilisearch removal: no secondary index to delete from

    total_elapsed = time.time() - total_start
    logger.info(f"✅ Delete completed: {deleted_count} record(s) for {identifier} in {total_elapsed:.3f}s")
    
    return {
        "message": f"Successfully deleted {deleted_count} record(s) and associated files for {identifier}.",
        "rows_deleted": deleted_count
    }

@router.get("/sendables/download")
async def download_sendable(
    companyId: str,
    filename:  str = Query(...)
):
    logger.debug(f"📥 Download request: {filename} for company {companyId}")
    
    path = f"uploads/{companyId}/sendables/{filename}"
    blob = bucket.blob(path)
    
    if not blob.exists():
        logger.warning(f"⚠️  File not found in GCS: {path}")
        raise HTTPException(status_code=404, detail="Файл не найден в GCS")

    signed_url = blob.generate_signed_url(expiration=15 * 60)  # 15 минут
    logger.info(f"✅ Generated signed URL for: {filename}")
    
    return {"url": signed_url}

@router.get("/sendables/list")
async def list_sendables(companyId: str):
    start_time = time.time()
    
    _debug_request_info("GET /sendables/list", companyId)
    
    collection = get_company_sendable_table(companyId)
    logger.debug(f"📦 Collection: {collection}")
    
    results = []
    cursor = None
    batch_count = 0
    
    logger.debug("📜 Scrolling through all sendables...")
    while True:
        batch, cursor = qdrant.scroll(
            collection_name=collection, 
            with_payload=True, 
            limit=1000, 
            offset=cursor
        )
        batch_count += 1
        
        if not batch:
            break
            
        logger.debug(f"   Batch {batch_count}: {len(batch)} records")
        
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
    
    elapsed = time.time() - start_time
    logger.info(f"✅ List completed: {len(results)} sendables in {elapsed:.3f}s ({batch_count} batches)")
    
    return {"sendables": results}



import os
import io
import json
import base64
from typing import List
import hashlib
import urllib.parse

from fastapi import FastAPI, Form, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper

from google.oauth2 import service_account
from google.cloud import storage

from typing import List, Optional
import pyarrow as pa
from lancedb.pydantic import LanceModel, Vector

load_dotenv()

# ------------------------------------------------------------------------------
# Глобальные настройки
# ------------------------------------------------------------------------------
client = OpenAI()  # берет OPENAI_API_KEY из .env
tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = 8191
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
    file_type: str | None
    filename: str | None
    url: str | None

class SendableFile(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: SendableMetadata

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

# @app.post("/{companyId}/delete-all") # для удаления всех таблиц компании
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
async def process_document(companyId: str, file: UploadFile = File(...)):
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
        # 2) Save file locally first
        upload_dir = os.path.join("uploads", companyId, "temp")
        os.makedirs(upload_dir, exist_ok=True)
        temp_path = os.path.join(upload_dir, safe_filename)
        
        content = await file.read()
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
        try:
            result = converter.convert(source=temp_path)
            document = result.document
            
            # Optional: Export markdown for debugging
            markdown_output = document.export_to_markdown()
            print("Markdown output:", markdown_output)

            # 5) Create chunks
            try:
                print("Starting chunking process...")
                chunker = HybridChunker(
                    tokenizer=tokenizer,
                    max_tokens=MAX_TOKENS,
                    merge_peers=True
                )
                print("Created chunker successfully")
                
                try:
                    chunks = list(chunker.chunk(dl_doc=document))
                    print(f"Created {len(chunks)} chunks from document")
                except Exception as chunk_error:
                    print(f"Error during chunking: {str(chunk_error)}")
                    import traceback
                    print(f"Chunking traceback: {traceback.format_exc()}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error during chunking: {str(chunk_error)}\nTraceback: {traceback.format_exc()}"
                    )
                
                if not chunks:
                    raise ValueError("No chunks were created from the document")

                # 6) Prepare chunks for storage
                to_store = []
                for i, chunk in enumerate(chunks):
                    try:
                        page_nums = sorted({prov.page_no for item in chunk.meta.doc_items for prov in item.prov}) or []
                        # Convert page numbers to JSON string
                        page_nums_str = json.dumps(page_nums) if page_nums else None
                        
                        to_store.append({
                            "text": chunk.text,
                            "metadata": {
                                "filename": safe_filename,
                                "page_numbers": page_nums_str,  # Store as JSON string
                                "title": chunk.meta.headings[0] if chunk.meta.headings else None,
                                "url": gcs_url,
                            },
                        })
                        print(f"Processed chunk {i+1}/{len(chunks)}")
                    except Exception as chunk_error:
                        print(f"Error processing chunk {i+1}: {str(chunk_error)}")
                        import traceback
                        print(f"Chunk processing traceback: {traceback.format_exc()}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Error processing chunk {i+1}: {str(chunk_error)}\nTraceback: {traceback.format_exc()}"
                        )
                if not to_store:
                    raise ValueError("No chunks were prepared for storage")
                print(f"Preparing to store {len(to_store)} chunks. First few items:")
                for i, item_to_store in enumerate(to_store[:3]): # Print first 3 items
                    print(f"Item {i}:")
                    print(f"  Text: {item_to_store['text'][:50]}...") # First 50 chars
                    print(f"  Metadata: {item_to_store['metadata']}")
                    # Specifically check page_numbers type and value
                    page_numbers_val = item_to_store['metadata'].get('page_numbers')
                    print(f"  Metadata.page_numbers: {page_numbers_val} (type: {type(page_numbers_val)})")

                # 7) Store in LanceDB
                try:
                    table.add(to_store)
                    print(f"Added {len(to_store)} chunks to LanceDB")
                except Exception as storage_error:
                    print(f"Error storing chunks in LanceDB: {str(storage_error)}")
                    import traceback
                    print(f"Storage traceback: {traceback.format_exc()}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error storing chunks in LanceDB: {str(storage_error)}\nTraceback: {traceback.format_exc()}"
                    )

                return {
                    "message": "Document processed and embeddings stored successfully.",
                    "row_count": table.count_rows(),
                    "url": gcs_url
                }

            except Exception as chunking_error:
                print(f"Chunking error: {str(chunking_error)}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error during document chunking: {str(chunking_error)}\nTraceback: {traceback.format_exc()}"
                )

        except Exception as e:
            print(f"Document processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
        finally:
            # Clean up temp file
            try:
                os.remove(temp_path)
                print(f"Cleaned up temporary file: {temp_path}")
            except Exception as cleanup_error:
                print(f"Warning: Failed to clean up temporary file: {cleanup_error}")

    except Exception as e:
        print(f"General error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/companies/{companyId}/search")
async def search_documents(companyId: str, query: str = Query(...), limit: int = Query(5)):
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

    # 2) Search using the vector
    try:
        df = table.search(query_vec).limit(limit).to_pandas()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    # 3) Format results
    results = []
    for _, r in df.iterrows():
        md = r["metadata"]
        # Parse page numbers from JSON string if present
        page_numbers = json.loads(md["page_numbers"]) if md["page_numbers"] else None
        
        results.append({
            "text":         r["text"],
            "filename":     md["filename"],
            "page_numbers": page_numbers,  # Now properly parsed from JSON
            "title":        md["title"],
            "url":          md["url"],
            "score":        float(r.get("_distance", 0))  # Add relevance score
        })
    return {"results": results}

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

    # 1) Проверяем, не загружали ли уже этот файл
    #    — вытягиваем все записи в DataFrame и фильтруем по metadata.filename
    df = table.to_pandas()
    if df["metadata"].apply(lambda md: md.get("filename") == safe_name).any():
        raise HTTPException(
            status_code=400,
            detail=f"Sendable '{safe_name}' уже обработан для компании {companyId}"
        )

    # 2) Читаем содержимое и заливаем в GCS
    content = await file.read()
    gcs_url = upload_to_gcs(
        content,
        companyId,
        "sendables",
        safe_name,
        file.content_type or "application/octet-stream"
    )

    # 3) Генерим embedding описания
    emb_resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=description
    )
    embedding = emb_resp.data[0].embedding

    # 4) Готовим запись и сохраняем в LanceDB
    record = {
        "text":    description,
        "vector":  embedding,
        "metadata": {
            "file_type": file.content_type or os.path.splitext(safe_name)[1],
            "filename":  safe_name,
            "url":       gcs_url
        }
    }
    table.add([record])

    return {
        "message":   "Sendable файл обработан и сохранён.",
        "row_count": table.count_rows(),
        "url" : gcs_url
    }

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
            "text":      row["text"],
            "filename":  md["filename"],
            "file_type": md["file_type"],
            "url":       md["url"],
            "score":     float(row.get("_distance", 0))
        })

    return {"results": results}

@app.delete("/companies/{companyId}/delete-sendable")
async def delete_sendable(
    companyId: str,
    filename:  str = Query(...)
):
    # 0) Подготовка
    safe_name = safe_decode_filename(filename)
    table     = get_company_sendable_table(companyId)

    # 1) Считаем, сколько записей соответствует filename
    df = table.to_pandas()
    deleted_count = int(
        df["metadata"]
          .apply(lambda md: md.get("filename") == safe_name)
          .sum()
    )

    # 2) Удаляем из LanceDB
    filter_q = f"metadata.filename = '{safe_name}'"
    table.delete(filter_q)

    # 3) Удаляем blob в GCS
    path = f"uploads/{companyId}/sendables/{safe_name}"
    bucket.blob(path).delete()

    return {
        "message":      f"Удалено {deleted_count} записей и файл из GCS",
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
    """
    Returns a list of 'sendable' records from the database (sendable_files_{companyId}),
    including each file's name and description.
    """
    # 1) Open the LanceDB table for this company
    sendable_table = get_company_sendable_table(companyId)
    
    # 2) Convert all rows to a DataFrame (or you can do a search().limit(...) if needed)
    df = sendable_table.to_pandas()
    
    # 3) Build a serializable list of {filename, description, ...}
    results = []
    for _, row in df.iterrows():
        metadata = row['metadata']
        # 'text' holds the user's description,
        # 'metadata.filename' is the original filename,
        # 'metadata.url' is the file's public URL (if you assigned it),
        # 'metadata.file_type' might be ".pdf", ".png", etc.
        result_item = {
            "filename": metadata["filename"],
            "description": row["text"],
            "file_type": metadata["file_type"],
            "url": metadata["url"]
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



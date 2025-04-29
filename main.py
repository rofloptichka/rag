import os
import io
import json
import base64
from typing import List
import hashlib

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
    filename: str | None
    page_numbers: List[int] | None
    title: str | None
    url:         str

class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadata

class SendableMetadata(LanceModel):
    file_type: str | None
    filename: str | None
    url: str | None

class SendableFile(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: SendableMetadata


# ------------------------------------------------------------------------------
# Хелпер-функции: открытие (или создание) таблиц на компанию
# ------------------------------------------------------------------------------
def get_company_doc_table(company_id: str):
    """
    Возвращает таблицу для чанкнутых документов docling_{companyId},
    создаёт новую, если не найдена.
    """
    table_name = f"docling_{company_id}"
    try:
        table = db.open_table(table_name)
        print(f"Opened existing table '{table_name}'.")
    except ValueError:
        table = db.create_table(table_name, schema=Chunks, mode="create")
        print(f"Created new table '{table_name}'.")
    return table

def get_company_sendable_table(company_id: str):
    """
    Возвращает таблицу для «sendable-файлов» sendable_files_{companyId},
    создаёт новую, если не найдена.
    """
    table_name = f"sendable_files_{company_id}"
    try:
        table = db.open_table(table_name)
        print(f"Opened existing table '{table_name}'.")
    except ValueError:
        table = db.create_table(table_name, schema=SendableFile, mode="create")
        print(f"Created new table '{table_name}'.")
    return table

def safe_decode_filename(filename: str) -> str:
    """Fixes improperly decoded filenames by attempting Latin-1 to UTF-8 conversion."""
    try:
        return filename.encode("latin1").decode("utf-8")
    except UnicodeDecodeError:
        return filename  # If decoding fails, return the original
    

def upload_to_gcs(
    content: bytes,
    company_id: str,
    folder: str,
    filename: str,
    content_type: str
) -> str:
    path = f"uploads/{company_id}/{folder}/{filename}"
    blob = bucket.blob(path)
    # Передаём правильный MIME-тип, чтобы PDF, картинки и т.п. рендерились корректно
    blob.upload_from_string(content, content_type=content_type)
    return f"gs://{bucket.name}/{path}"


# ------------------------------------------------------------------------------
# Эндпоинты для документов (docling_{companyId})
# ------------------------------------------------------------------------------
@app.post("/companies/{companyId}/process-document")
async def process_document(companyId: str, file: UploadFile = File(...)):
    safe_filename = safe_decode_filename(file.filename)

    # 1) Проверка на дублирование по имени файла в БД
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

    # 2) Читаем в память и сразу заливаем в GCS
    content = await file.read()
    gcs_url = upload_to_gcs(content, companyId, "documents", safe_filename)

    # 3) Прогон через Docling без промежуточного файла на диске
    try:
        result   = converter.convert(file=io.BytesIO(content))
        document = result.document
        chunker  = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)
        chunks   = list(chunker.chunk(dl_doc=document))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document conversion failed: {e}")

    # 4) Формируем записи для LanceDB
    to_store = []
    for chunk in chunks:
        page_nums = sorted({prov.page_no for item in chunk.meta.doc_items for prov in item.prov}) or None
        to_store.append({
            "text": chunk.text,
            "metadata": {
                "filename":    safe_filename,
                "page_numbers": page_nums,
                "title":       chunk.meta.headings[0] if chunk.meta.headings else None,
                "url":         gcs_url,
            },
        })

    # 5) Сохраняем в таблицу
    table.add(to_store)
    return {
        "message": "Document processed and embeddings stored successfully.",
        "row_count": table.count_rows(),
        "url": gcs_url
        }



@app.get("/companies/{companyId}/search")
async def search_documents(companyId: str, query: str = Query(...), limit: int = Query(5)):
    table = get_company_doc_table(companyId)
    try:
        df = table.search(query=query).limit(limit).to_pandas()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    results = []
    for _, r in df.iterrows():
        md = r["metadata"]
        results.append({
            "text":         r["text"],
            "filename":     md["filename"],
            "page_numbers": md["page_numbers"],
            "title":        md["title"],
            "url":          md["url"],
        })
    return {"results": results}


@app.delete("/companies/{companyId}/delete-document")
async def delete_document(companyId: str, filename: str = Query(...)):
    table = get_company_doc_table(companyId)

    # 1) Считаем, сколько записей соответствует filename
    df = table.to_pandas()
    deleted = int(
        df["metadata"]
          .apply(lambda md: md.get("filename") == filename)
          .sum()
    )

    # 2) Удаляем из LanceDB
    q = f"metadata.filename = '{filename}'"
    table.delete(q)

    # 3) Удаляем blob в GCS
    path = f"uploads/{companyId}/documents/{filename}"
    bucket.blob(path).delete()

    return {
        "message":    f"Deleted {deleted} embeddings and removed file from GCS.",
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
    upload_dir = os.path.join("uploads", companyId, "documents")
    if not os.path.exists(upload_dir):
        return {"files": []}

    filenames = os.listdir(upload_dir)
    return {"files": filenames}


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
        # 'text' holds the user’s description,
        # 'metadata.filename' is the original filename,
        # 'metadata.url' is the file’s public URL (if you assigned it),
        # 'metadata.file_type' might be ".pdf", ".png", etc.
        result_item = {
            "filename": metadata["filename"],
            "description": row["text"],
            "file_type": metadata["file_type"],
            "url": metadata["url"]
        }
        results.append(result_item)

    return {"sendables": results}



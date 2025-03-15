from fastapi import FastAPI, UploadFile, File, Query, Form, HTTPException
from fastapi.responses import FileResponse
import os
import lancedb
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from typing import List
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper
import json
from pydantic import ValidationError
import base64

load_dotenv()

# ------------------------------------------------------------------------------
# Глобальные настройки
# ------------------------------------------------------------------------------
client = OpenAI()  # берет OPENAI_API_KEY из .env
tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = 8191
converter = DocumentConverter()

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

# ------------------------------------------------------------------------------
# Эндпоинты для документов (docling_{companyId})
# ------------------------------------------------------------------------------
@app.post("/companies/{companyId}/process-document")
async def process_document(companyId: str, file: UploadFile = File(...)):
    """
    Uploads and processes a document for the given companyId (using Docling).
    The resulting chunks are stored in the table docling_{companyId}.
    """
    upload_dir = os.path.join("uploads", companyId, "documents")
    os.makedirs(upload_dir, exist_ok=True)

    # Decode the filename properly
    safe_filename = safe_decode_filename(file.filename)

    file_location = os.path.join(upload_dir, safe_filename)

    # Prevent overwriting existing files
    if os.path.exists(file_location):
        raise HTTPException(
            status_code=400,
            detail=(
                f"File '{safe_filename}' already exists for company '{companyId}'. "
                "Please delete the old file or rename the new one."
            )
        )

    # Save the file
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Process the document
    try:
        result = converter.convert(file_location)
        document = result.document
        chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)
        chunks = list(chunker.chunk(dl_doc=document))
    except Exception as e:
        os.remove(file_location)  # Clean up if processing fails
        raise HTTPException(status_code=500, detail=f"Document conversion failed: {e}")

    # Prepare chunks for storage
    processed_chunks = []
    for chunk in chunks:
        page_nums = sorted({prov.page_no for item in chunk.meta.doc_items for prov in item.prov}) or None
        processed_chunks.append({
            "text": chunk.text,
            "metadata": {
                "filename": safe_filename,
                "page_numbers": page_nums,
                "title": chunk.meta.headings[0] if chunk.meta.headings else None,
            },
        })

    # Store chunked data in LanceDB
    company_table = get_company_doc_table(companyId)
    company_table.add(processed_chunks)

    row_count = company_table.count_rows()
    return {"message": "Document processed and embeddings stored successfully.", "row_count": row_count}


@app.get("/companies/{companyId}/search")
async def search_documents(companyId: str, query: str = Query(...), limit: int = Query(5)):
    """
    Поиск (vector search) по таблице docling_{companyId}.
    """
    try:
        company_table = get_company_doc_table(companyId)
        result = company_table.search(query=query).limit(limit)
        records = result.to_pandas().to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

    # Приводим к JSON-сериализуемому формату
    def make_serializable(value):
        try:
            json.dumps(value)
            return value
        except Exception:
            if hasattr(value, "dict"):
                return value.dict()
            return str(value)

    serializable_records = []
    for record in records:
        new_record = {k: make_serializable(v) for k, v in record.items()}
        serializable_records.append(new_record)

    return {"results": serializable_records}


@app.delete("/companies/{companyId}/delete-document")
async def delete_document(companyId: str, filename: str = Query(...)):
    """
    Удаляем (локальный файл + связанные embeddings) для указанной компании.
    """
    file_path = os.path.join("uploads", companyId, "documents", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    os.remove(file_path)

    company_table = get_company_doc_table(companyId)
    query_str = f"metadata.filename = '{filename}'"
    matching_rows = company_table.search().where(query_str).to_pandas()

    if matching_rows.empty:
        return {
            "message": f"File '{filename}' removed from disk, but no embeddings found in LanceDB.",
            "rows_deleted": 0
        }

    company_table.delete(query_str)
    row_count = len(matching_rows)
    return {
        "message": f"File '{filename}' and its {row_count} embeddings were removed successfully.",
        "rows_deleted": row_count
    }

@app.get("/companies/{companyId}/documents/download")
async def download_document(companyId: str, filename: str = Query(...)):
    """
    Эндпоинт для скачивания файла, который лежит
    в uploads/{companyId}/documents/<filename>
    """
    file_path = os.path.join("uploads", companyId, "documents", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # FileResponse вернёт сам файл пользователю.
    return FileResponse(
        file_path,
        media_type="application/octet-stream",  # или другой mime-тип, если хотите
        filename=filename                       # имя скачиваемого файла
    )


# ------------------------------------------------------------------------------
# Эндпоинты для «sendable-files» (sendable_files_{companyId})
# ------------------------------------------------------------------------------
@app.post("/companies/{companyId}/process-sendable-file")
async def process_sendable_file(companyId: str, file: UploadFile = File(...), description: str = Form(...)):
    """
    Processes small "sendable" files (e.g., images, short texts) and stores them in sendable_files_{companyId}.
    """
    upload_dir = os.path.join("uploads", companyId, "sendable")
    os.makedirs(upload_dir, exist_ok=True)

    # Decode the filename properly
    safe_filename = safe_decode_filename(file.filename)

    file_location = os.path.join(upload_dir, safe_filename)

    # Prevent duplicate filenames
    if os.path.exists(file_location):
        raise HTTPException(
            status_code=400,
            detail=(
                f"File '{safe_filename}' already exists for company '{companyId}'. "
                "Please delete the old file or rename the new one."
            )
        )

    # Save file
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Construct file URL
    file_url = f"https://threegis.org/uploads/{companyId}/sendable/{safe_filename}"

    metadata = SendableMetadata(
        file_type=os.path.splitext(safe_filename)[1].lower(),
        filename=safe_filename,
        url=file_url
    )

    try:
        # Generate embedding for description
        embedding_response = client.embeddings.create(
            model="text-embedding-3-large",
            input=description
        )
        embedding = embedding_response.data[0].embedding

        data = {
            "text": description,
            "vector": embedding,
            "metadata": metadata.dict()
        }

        sendable_table = get_company_sendable_table(companyId)
        sendable_table.add([data])
        row_count = sendable_table.count_rows()
        return {"message": "Sendable file processed and stored successfully.", "row_count": row_count}

    except Exception as e:
        # Clean up if an error occurs
        if os.path.exists(file_location):
            os.remove(file_location)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/companies/{companyId}/search-sendable")
async def search_sendable(companyId: str, query: str = Query(...), limit: int = Query(5)):
    """
    Векторный поиск по sendable_files_{companyId}.
    """
    try:
        embedding_response = client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        query_embedding = embedding_response.data[0].embedding

        sendable_table = get_company_sendable_table(companyId)
        result = sendable_table.search(query_embedding).limit(limit).to_pandas()

        serializable_records = []
        for _, record in result.iterrows():
            metadata = record['metadata']
            new_record = {
                'text': record['text'],
                'metadata': {
                    'file_type': metadata['file_type'],
                    'filename': metadata['filename'],
                    'url': metadata['url']
                },
                'score': float(record.get('_distance', 0))
            }
            serializable_records.append(new_record)

        return {"results": serializable_records}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


@app.delete("/companies/{companyId}/delete-sendable")
async def delete_sendable(companyId: str, filename: str = Query(...)):
    """
    Удаляем «sendable»-файл и его запись из LanceDB (таблица sendable_files_{companyId}).
    """
    file_path = os.path.join("uploads", companyId, "sendable", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    os.remove(file_path)
    sendable_table = get_company_sendable_table(companyId)
    query_str = f"metadata.filename = '{filename}'"
    matching_rows = sendable_table.search().where(query_str).to_pandas()

    if matching_rows.empty:
        return {
            "message": f"Sendable file '{filename}' removed from disk, but no record found in LanceDB.",
            "rows_deleted": 0
        }

    sendable_table.delete(query_str)
    row_count = len(matching_rows)
    return {
        "message": f"Sendable file '{filename}' and its record were removed successfully.",
        "rows_deleted": row_count
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

@app.get("/companies/{companyId}/sendables/download")
async def download_sendable(companyId: str, filename: str = Query(...)):
    file_path = os.path.join("uploads", companyId, "sendable", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=filename
    )



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



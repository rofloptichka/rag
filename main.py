from fastapi import FastAPI, UploadFile, File, Query, HTTPException
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

load_dotenv()

# Initialize OpenAI client using environment variable for API key
client = OpenAI()

# Initialize our custom tokenizer for OpenAI
tokenizer = OpenAITokenizerWrapper()  
MAX_TOKENS = 8191  # Maximum tokens for text-embedding-3-large

# Initialize the DocumentConverter from Docling
converter = DocumentConverter()

# Create FastAPI application
app = FastAPI()

# Connect to LanceDB (assumes "data/lancedb" folder exists or will be created)
db = lancedb.connect("data/lancedb")

# Get the OpenAI embedding function from LanceDB's registry
func = get_registry().get("openai").create(name="text-embedding-3-large")

# --------------------------------------------------------------
# Define Schema Classes (at module level)
# --------------------------------------------------------------

class ChunkMetadata(LanceModel):
    """
    Simplified metadata schema.
    Fields must be in alphabetical order.
    """
    filename: str | None
    page_numbers: List[int] | None
    title: str | None

class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadata

# --------------------------------------------------------------
# Helper: Get or create a table for a specific company
# --------------------------------------------------------------

def get_company_table(company_id: str):
    table_name = f"docling_{company_id}"
    try:
        table = db.open_table(table_name)
        print(f"Opened existing table '{table_name}'.")
    except Exception as e:
        # If the table doesn't exist, create a new one for this company.
        table = db.create_table(table_name, schema=Chunks, mode="create")
        print(f"Created new table '{table_name}'.")
    return table

# --------------------------------------------------------------
# Document Processing Endpoint for a Company
# --------------------------------------------------------------

@app.post("/companies/{companyId}/documents")
async def process_document(
    companyId: str,
    file: UploadFile = File(...)
):
    # Create a directory for the company's uploads
    upload_dir = os.path.join("uploads", "documents", companyId)
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save the uploaded file locally in the company's folder
    file_location = os.path.join(upload_dir, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Process the document using Docling
    try:
        result = converter.convert(file_location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document conversion failed: {e}")
    
    document = result.document

    # (Optional) Export markdown for debugging
    markdown_output = document.export_to_markdown()
    print("Markdown output:", markdown_output)

    # Apply hybrid chunking to split the document into chunks
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        merge_peers=True,
    )
    chunk_iter = chunker.chunk(dl_doc=document)
    chunks = list(chunk_iter)

    # Prepare processed chunks for storage, including metadata
    processed_chunks = [
        {
            "text": chunk.text,
            "metadata": {
                "filename": chunk.meta.origin.filename,
                "page_numbers": sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    )
                ) or None,
                "title": chunk.meta.headings[0] if chunk.meta.headings else None,
            },
        }
        for chunk in chunks
    ]

    # Get (or create) the table for the given company and add the chunks
    table = get_company_table(companyId)
    table.add(processed_chunks)
    
    row_count = table.count_rows()
    return {"message": "Document processed and embeddings stored successfully.", "row_count": row_count}

# --------------------------------------------------------------
# Search Endpoint for a Company
# --------------------------------------------------------------

@app.get("/companies/{companyId}/documents")
async def search(
    companyId: str,
    query: str = Query(..., description="Search query"),
    limit: int = Query(5, description="Limit the number of results")
):
    try:
        table = get_company_table(companyId)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error opening table for company {companyId}: {e}")

    result = table.search(query=query).limit(limit)
    # Convert the results to a list of dictionaries using pandas
    records = result.to_pandas().to_dict(orient="records")
    
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

# --------------------------------------------------------------
# Delete Document Endpoint for a Company
# --------------------------------------------------------------

@app.delete("/companies/{companyId}/documents")
async def delete_document(
    companyId: str,
    filename: str = Query(..., description="Name of the file to delete")
):
    # 1) Remove the file from local storage
    file_path = os.path.join("uploads", "documents", companyId, filename)
    if not os.path.exists(file_path):
        print(f"File not found on disk: {file_path}")
        raise HTTPException(status_code=404, detail="File not found on disk")
    os.remove(file_path)
    print(f"Removed file: {file_path}")

    # 2) Remove associated chunks from LanceDB for the company
    table = get_company_table(companyId)
    query_str = f"metadata.filename = '{filename}'"
    matching_rows = table.search().where(query_str).to_pandas()
    
    if matching_rows.empty:
        print(f"No matching rows found for filename='{filename}' in LanceDB.")
        return {
            "message": f"Document '{filename}' removed from disk, but no embeddings found in LanceDB.",
            "rows_deleted": 0
        }

    table.delete(query_str)
    row_count = len(matching_rows)
    print(f"Deleted {row_count} rows for filename='{filename}' from LanceDB.")
    
    return {
        "message": f"Document '{filename}' and its {row_count} embeddings were removed successfully.",
        "rows_deleted": row_count
    }

# --------------------------------------------------------------
# List Documents Endpoint for a Company
# --------------------------------------------------------------

@app.get("/companies/{companyId}/documents/list")
async def list_documents(companyId: str):
    # Build the path to the company's uploads folder
    upload_dir = os.path.join("uploads", "documents", companyId)
    if not os.path.exists(upload_dir):
        return {"files": []}
    
    # List all filenames in the directory
    filenames = os.listdir(upload_dir)
    return {"files": filenames}

# --------------------------------------------------------------
# Download Document Endpoint for a Company (optional)
# --------------------------------------------------------------

@app.get("/companies/{companyId}/documents/download")
async def download_document(
    companyId: str,
    filename: str = Query(..., description="Name of the file to download")
):
    file_path = os.path.join("uploads", "documents", companyId, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)

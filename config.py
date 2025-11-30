# config.py
import os
import redis
import json
import base64
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from google.oauth2 import service_account
from google.cloud import storage
import cloudinary
from utils.tokenizer import OpenAITokenizerWrapper

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import google.generativeai as genai


# Загружаем переменные окружения
load_dotenv()

# --- Клиенты и Глобальные Настройки ---

# OpenAI (for reranking and other tasks)
from openai import AsyncOpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "10"))
client_fast = client.with_options(timeout=OPENAI_TIMEOUT)
client_async = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT)

# Gemini Embeddings Configuration
# gemini-embedding-001: Best multilingual model, supports Russian/Kazakh
# 3072 dimensions, faster than OpenAI, excellent quality
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # or GEMINI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
EMBED_DIMS = int(os.getenv("EMBED_DIMS", "768"))  # Use 768 for speed, up to 3072 for max quality
EMBED_TASK_TYPE = os.getenv("EMBED_TASK_TYPE", "RETRIEVAL_DOCUMENT")  # or RETRIEVAL_QUERY for queries

# Google Cloud Storage
b64_key = os.environ["GOOGLE_CLOUD_KEY"]
decoded = base64.b64decode(b64_key)
sa_info = json.loads(decoded)
creds = service_account.Credentials.from_service_account_info(sa_info)
storage_client = storage.Client(credentials=creds, project=sa_info["project_id"])
bucket = storage_client.bucket(os.environ["GCS_BUCKET_NAME"])

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True, timeout=60)

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
try:
    r = redis.from_url(REDIS_URL, decode_responses=True)
    r.ping()
except Exception:
    r = None # Обработка ошибок остается здесь

# Cloudinary
cloudinary.config(
  cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
  api_key = os.getenv("CLOUDINARY_API_KEY"),
  api_secret = os.getenv("CLOUDINARY_API_SECRET"),
  secure=True
)

# --- Feature Flags и другие настройки ---
def _parse_bool_env(value: str | None, default: bool) -> bool:
    if value is None: return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

# Docling configuration
converter = DocumentConverter()
tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = int(os.getenv("HYBRID_CHUNK_MAX_TOKENS", "512"))
chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)


# RAG configuration
RAG_USE_LLM_UNDERSTAND_DEFAULT = _parse_bool_env(os.getenv("RAG_USE_LLM_UNDERSTAND"), False)
RAG_NATURAL_CHUNKING_DEFAULT   = _parse_bool_env(os.getenv("RAG_NATURAL_CHUNKING"), True)
RAG_SOFT_MAX_TOKENS            = int(os.getenv("RAG_SOFT_MAX_TOKENS", "300"))
RAG_HARD_MAX_TOKENS            = int(os.getenv("RAG_HARD_MAX_TOKENS", "700"))

# OpenAI reranker configuration
OPENAI_RERANKER_ENABLED        = _parse_bool_env(os.getenv("OPENAI_RERANKER_ENABLED"), False)
OPENAI_RERANKER_MODEL          = os.getenv("OPENAI_RERANKER_MODEL", "gpt-4.1-nano")
OPENAI_RERANKER_TOP_K          = int(os.getenv("OPENAI_RERANKER_TOP_K", "20"))
OPENAI_RERANKER_TIMEOUT        = int(os.getenv("OPENAI_RERANKER_TIMEOUT", "15"))

# Jina reranker configuration
JINA_RERANKER_ENABLED          = _parse_bool_env(os.getenv("JINA_RERANKER_ENABLED"), False)
JINA_API_KEY                   = os.getenv("JINA_API_KEY")
JINA_RERANKER_MODEL            = os.getenv("JINA_RERANKER_MODEL", "jina-reranker-v2-base-multilingual")
JINA_RERANKER_ENDPOINT         = os.getenv("JINA_RERANKER_ENDPOINT", "https://api.jina.ai/v1/rerank")
JINA_RERANKER_TIMEOUT          = int(os.getenv("JINA_RERANKER_TIMEOUT", "15"))


# main.py
import time
import logging
from fastapi import FastAPI, Request

# Импортируем наши роутеры
from routers import documents, sendables

# Настраиваем логирование (можно вынести в отдельный logging_config.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Создаем приложение FastAPI
app = FastAPI(
    title="Docling RAG API",
    description="API для обработки и поиска документов.",
    version="1.0.0"
)

# Подключаем middleware (он должен остаться здесь, т.к. применяется ко всему приложению)
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-ms"] = f"{process_time_ms:.2f}"
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time_ms:.2f} ms")
    return response

# === Подключение роутеров ===
# Вот здесь происходит магия. Мы "включаем" все эндпоинты из наших файлов.
app.include_router(documents.router)
app.include_router(sendables.router)

# Можно добавить корневой эндпоинт для проверки, что API работает
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the Docling RAG API!"}
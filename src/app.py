from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from config import logger, settings
from schemas.faiss import SearchResult, SearchRequest
import uvicorn
from manager import manager
import os

app = FastAPI(
    title="VectorDB API",
    description="API для работы с векторной базой данных",
    version="0.1.0"
)

@app.post("/search/", response_model=List[SearchResult], summary="Поиск документов")
async def search_documents(request: SearchRequest):
    """
    Ищет документы, похожие на заданный запрос.
    """
    try:
        results = manager.similarity_search(request.query, k=request.k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/", summary="Проверка здоровья сервиса")
async def health_check() -> Dict[str, Any]:
    """
    Проверка работоспособности сервиса.
    """
    return {
        "status": "OK",
    }


if __name__ == "__main__":
    try:
        uvicorn.run(
            app,
            host=settings.app_host,
            port=settings.app_port,
            log_config="src/logs/log_config.json",
            use_colors=True,
            log_level="info",
            loop="asyncio"
        )
    except Exception as e:
        logger.error(f"Error launch app: {e}")

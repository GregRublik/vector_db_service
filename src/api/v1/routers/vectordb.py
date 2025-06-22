from fastapi import HTTPException
from typing import List, Dict, Any
from api.v1.schemas.faiss import SearchResult, SearchRequest
from manager import manager
from fastapi import APIRouter

vectordb = APIRouter()


@vectordb.post("/search/", response_model=List[SearchResult], summary="Поиск документов")
async def search_documents(request: SearchRequest):
    """
    Ищет документы, похожие на заданный запрос.
    """
    try:
        results = manager.similarity_search(request.query, k=request.k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@vectordb.get("/health/", summary="Проверка здоровья сервиса")
async def health_check() -> Dict[str]:
    """
    Проверка работоспособности сервиса.
    """
    return {
        "status": "OK",
    }

from fastapi import HTTPException
from typing import List, Dict, Any
from api.v1.schemas.faiss import SearchResult, SearchRequest
from services.vectordb.manager import manager
from services.vectordb.retriever import retriever
from fastapi import APIRouter

vectordb = APIRouter()


@vectordb.post("/search/", response_model=List[SearchResult], summary="Поиск документов")
async def search_documents(request: SearchRequest):
    """
    Ищет документы, похожие на заданный запрос.
    """
    try:
        results = manager.similarity_search(request.query, k=request.k)

        results_reranked = retriever.process_search_results(
            query=request.query,
            search_results=results,
            rerank=True,
            top_k=3
        )

        print(f"retriever result - {results_reranked}")
        # return results_reranked

        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results_reranked]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@vectordb.get("/health/", summary="Проверка здоровья сервиса")
async def health_check() -> Dict[str, Any]:
    """
    Проверка работоспособности сервиса.
    """
    return {
        "status": "OK",
    }

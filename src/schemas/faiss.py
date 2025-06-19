from pydantic import BaseModel
from typing import Optional


class SearchRequest(BaseModel):
    query: str
    k: int = 3

class SearchResult(BaseModel):
    content: str
    metadata: dict
    score: Optional[float] = None

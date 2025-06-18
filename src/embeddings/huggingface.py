from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from embeddings.base import BaseEmbedding


class HuggingFaceEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def get_embedding(self):
        # Сначала создаем SentenceTransformer с trust_remote_code
        model = SentenceTransformer(
            self.model_name,
            device=self.kwargs.get("model_kwargs", {}).get("device", "cpu"),
            trust_remote_code=True
        )

        # Затем создаем LangChain-совместимую обертку
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            encode_kwargs={
                "device": self.kwargs.get("model_kwargs", {}).get("device", "cpu"),
                "normalize_embeddings": True
            }
        )
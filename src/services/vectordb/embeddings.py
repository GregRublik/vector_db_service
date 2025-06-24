from abc import ABC, abstractmethod
from utils.decorators import measure_time
from config.settings import logger
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingFactory:
    """
    Фабрика эмбеддингов
    """

    @staticmethod
    @measure_time
    def create_embedding(embedding_type="huggingface", **kwargs):
        if embedding_type == "huggingface":
            logger.info(f"Проверка модели ...")
            return HuggingFaceEmbedding(**kwargs).get_embedding()
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")


class BaseEmbedding(ABC):
    @abstractmethod
    def get_embedding(self):
        pass


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

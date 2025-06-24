from services.vectordb.embeddings.huggingface import HuggingFaceEmbedding
from utils.decorators import measure_time
from config.settings import logger


class EmbeddingFactory:

    @staticmethod
    @measure_time
    def create_embedding(embedding_type="huggingface", **kwargs):
        if embedding_type == "huggingface":
            logger.info(f"Проверка модели ...")
            return HuggingFaceEmbedding(**kwargs).get_embedding()
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

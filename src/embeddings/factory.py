from embeddings.huggingface import HuggingFaceEmbedding
from utils.decorators import measure_time


class EmbeddingFactory:

    @staticmethod
    @measure_time
    def create_embedding(embedding_type="huggingface", **kwargs):
        if embedding_type == "huggingface":
            return HuggingFaceEmbedding(**kwargs).get_embedding()
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

from langchain_huggingface import HuggingFaceEmbeddings
from embeddings.base import BaseEmbedding


class HuggingFaceEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def get_embedding(self):
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            **self.kwargs
        )

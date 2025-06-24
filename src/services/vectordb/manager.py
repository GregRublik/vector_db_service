from typing import List, Literal
from langchain_core.documents import Document
from services.vectordb.embeddings.factory import EmbeddingFactory
from services.vectordb.vector_stores.faiss import FAISSStore
from services.vectordb.vector_stores.chroma import ChromaStore
from config import settings
from utils.docs_loader import load_markdown_documents

VectorStoreType = Literal["faiss", "chroma"]

class VectorDBManager:
    """
    Унифицированный менеджер для работы с векторными базами данных.

    Позволяет:
    - Инициализировать разные типы векторных хранилищ (FAISS/ChromaDB)
    - Работать с различными моделями эмбеддингов
    - Сохранять/загружать векторные индексы
    - Выполнять семантический поиск
    """
    def __init__(
            self,
            store_name: str,
            embedding_type: str = "huggingface",
            vector_store_type: VectorStoreType = "faiss",
            embedding_kwargs: dict = None
    ):
        """
        Инициализация менеджера.
        :param store_name: Уникальное имя хранилища/коллекции
        :param embedding_type: Тип модели эмбеддингов (huggingface)
        :param vector_store_type: Тип векторной БД (faiss/chroma)
        :param embedding_kwargs: Параметры для инициализации модели эмбеддингов
        """
        self.embedding = EmbeddingFactory.create_embedding(
            embedding_type,
            **(embedding_kwargs or {})
        )

        self.store_name = store_name
        self.vector_store_type = vector_store_type

        if vector_store_type == "faiss":
            self.vector_store = FAISSStore(
                self.embedding,
                str(settings.vector_store_dir),
                store_name
            )
        elif vector_store_type == "chroma":
            self.vector_store = ChromaStore(
                self.embedding,
                str(settings.vector_store_dir),
                store_name
            )
        else:
            raise ValueError(f"Unknown vector store type: {vector_store_type}")

    def add_documents(self, documents: List[Document]):
        self.vector_store.add_documents(documents)
        self.vector_store.save()

    def similarity_search(self, query: str, **kwargs):
        return self.vector_store.similarity_search(query, **kwargs)

    def load_store(self):
        self.vector_store.load()


# Инициализация с FAISS or Chroma
manager = VectorDBManager(
    store_name="faiss",
    embedding_type="huggingface",
    vector_store_type="faiss",
    embedding_kwargs={
        "model_name": settings.embedding_model,
        "model_kwargs": {"device": "cpu"},  # or "cuda"
    }
)

list_documents = {
    "Company Description - AVE Technologies.md": load_markdown_documents
}

for name, func in list_documents.items():
    # Загрузка документов
    documents = func(name)
    manager.add_documents(documents)

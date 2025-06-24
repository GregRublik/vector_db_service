from typing import List, Literal
from langchain_core.documents import Document
from services.vectordb.embeddings import EmbeddingFactory
from services.vectordb.vector_stores.faiss import FAISSStore
from services.vectordb.vector_stores.chroma import ChromaStore
from config.settings import settings
from utils.docs_loader import load_markdown_documents


class VectorDBManager:
    """
    Унифицированный менеджер для работы с векторными базами данных.

    Основные возможности:
    - Инициализация различных векторных хранилищ (FAISS, ChromaDB)
    - Работа с разными моделями эмбеддингов через EmbeddingFactory
    - Управление жизненным циклом векторных индексов
    - Семантический поиск по документам

    Атрибуты:
        embedding: Модель для создания векторных представлений текста
        store_name: Имя хранилища/коллекции
        vector_store_type: Тип используемого векторного хранилища
        vector_store: Экземпляр конкретного векторного хранилища
    """

    def __init__(
            self,
            store_name: str,
            embedding_type: str = "huggingface",
            vector_store_type: Literal["faiss", "chroma"] = "faiss",
            embedding_kwargs: dict = None
    ):
        """
        Инициализация менеджера векторной БД.

        Args:
            store_name: Уникальное имя хранилища/коллекции
            embedding_type: Тип модели эмбеддингов (по умолчанию "huggingface")
            vector_store_type: Тип векторного хранилища ("faiss" или "chroma")
            embedding_kwargs: Дополнительные параметры для инициализации модели эмбеддингов

        Raises:
            ValueError: Если указан неподдерживаемый тип векторного хранилища
        """
        # Инициализация модели эмбеддингов через фабрику
        self.embedding = EmbeddingFactory.create_embedding(
            embedding_type,
            **(embedding_kwargs or {})
        )

        self.store_name = store_name
        self.vector_store_type = vector_store_type

        # Инициализация конкретного векторного хранилища
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
        """
        Добавляет документы в векторное хранилище и сохраняет изменения.

        Args:
            documents: Список документов для добавления
        """
        self.vector_store.add_documents(documents)
        self.vector_store.save()

    def similarity_search(self, query: str, **kwargs):
        """
        Выполняет семантический поиск по векторному хранилищу.

        Args:
            query: Текстовый запрос для поиска
            **kwargs: Дополнительные параметры поиска:
                - k: Количество возвращаемых результатов
                - filter: Фильтры для поиска

        Returns:
            List[Document]: Список найденных документов, отсортированных по релевантности
        """
        return self.vector_store.similarity_search(query, **kwargs)

    def load_store(self):
        """
        Загружает ранее сохраненное векторное хранилище с диска.
        """
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

# Для загрузки существующего хранилища:
# manager.load_store()

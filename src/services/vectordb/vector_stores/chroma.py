from langchain_community.vectorstores import Chroma
from services.vectordb.vector_stores.base import VectorStoreInterface


class ChromaStore(VectorStoreInterface):
    """
    Конкретная реализация векторного хранилища на основе ChromaDB.

    Обеспечивает:
    - Хранение векторных представлений документов
    - Поиск по сходству
    - Сохранение/загрузку состояния
    """

    def __init__(self, embedding, store_path, collection_name):
        """
        Инициализирует хранилище ChromaDB.

        Args:
            embedding: Модель для создания эмбеддингов текста.
            store_path (str): Путь для сохранения/загрузки данных.
            collection_name (str): Название коллекции документов.
        """
        self.embedding = embedding
        self.store_path = store_path
        self.collection_name = collection_name
        self.store = None  # Инициализация происходит при первом добавлении или загрузке

    def add_documents(self, documents):
        """
        Добавляет документы в хранилище ChromaDB.

        Если хранилище не инициализировано, создает новую коллекцию.
        Если хранилище уже существует, добавляет документы в существующую коллекцию.

        Args:
            documents: Список документов для добавления.
        """
        if self.store is None:
            self.store = Chroma.from_documents(
                documents,
                self.embedding,
                persist_directory=self.store_path,
                collection_name=self.collection_name
            )
        else:
            self.store.add_documents(documents)

    def similarity_search(self, query, **kwargs):
        """
        Выполняет поиск документов, наиболее похожих на заданный запрос.

        Args:
            query (str): Текстовый запрос для поиска.
            **kwargs: Дополнительные параметры:
                - k (int): Количество возвращаемых документов
                - filter (dict): Фильтры для поиска

        Returns:
            List[Document]: Список документов, отсортированных по релевантности.
        """
        return self.store.similarity_search(query, **kwargs)

    def save(self):
        """
        Сохраняет текущее состояние хранилища на диск.
        Позволяет восстановить состояние при следующем запуске.
        """
        self.store.persist()

    def load(self):
        """
        Загружает ранее сохраненное хранилище с диска.
        Если хранилище не существует, инициализирует пустое состояние.
        """
        self.store = Chroma(
            persist_directory=self.store_path,
            embedding_function=self.embedding,
            collection_name=self.collection_name
        )

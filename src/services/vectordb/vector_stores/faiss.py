from langchain_community.vectorstores import FAISS
from services.vectordb.vector_stores.base import VectorStoreInterface
from utils.decorators import measure_time


class FAISSStore(VectorStoreInterface):
    """
        Конкретная реализация векторного хранилища на основе FAISS.

        Обеспечивает:
        - Хранение векторных представлений документов
        - Поиск по сходству
        - Сохранение/загрузку состояния
    """
    def __init__(self, embedding, store_path, store_name):
        """
        Инициализирует хранилище ChromaDB.

        Args:
            embedding: Модель для создания эмбеддингов текста.
            store_path (str): Путь для сохранения/загрузки данных.
            collection_name (str): Название коллекции документов.
        """
        self.embedding = embedding
        self.store_path = store_path
        self.store_name = store_name
        self.store = None

    @measure_time
    def add_documents(self, documents):
        """
        Добавляет документы в хранилище ChromaDB.

        Если хранилище не инициализировано, создает новую коллекцию.
        Если хранилище уже существует, добавляет документы в существующую коллекцию.

        Args:
            documents: Список документов для добавления.
        """
        if self.store is None:
            self.store = FAISS.from_documents(documents, self.embedding)
        else:
            self.store.add_documents(documents)

    @measure_time
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
        self.store.save_local(self.store_path, self.store_name)

    def load(self):
        """
        Загружает ранее сохраненное хранилище с диска.
        Если хранилище не существует, инициализирует пустое состояние.
        """
        self.store = FAISS.load_local(
            self.store_path,
            self.store_name,
            self.embedding,
            allow_dangerous_deserialization=True
        )

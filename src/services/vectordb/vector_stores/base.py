from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class VectorStoreInterface(ABC):
    """
    Абстрактный базовый интерфейс для работы с векторными хранилищами.

    Определяет стандартные методы, которые должны быть реализованы
    всеми конкретными классами векторных хранилищ.
    Обеспечивает единый интерфейс для различных реализаций векторных БД.
    """

    @abstractmethod
    def add_documents(self, documents: List[Document]):
        """
        Добавляет документы в векторное хранилище.

        Args:
            documents (List[Document]): Список документов для добавления.

        Raises:
            NotImplementedError: Если метод не реализован в дочернем классе.
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, **kwargs):
        """
        Выполняет поиск похожих документов по текстовому запросу.

        Args:
            query (str): Текстовый запрос для поиска.
            **kwargs: Дополнительные параметры поиска.

        Returns:
            List[Document]: Список найденных документов, отсортированных по релевантности.

        Raises:
            NotImplementedError: Если метод не реализован в дочернем классе.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Сохраняет текущее состояние векторного хранилища на диск.

        Raises:
            NotImplementedError: Если метод не реализован в дочернем классе.
        """
        pass

    @abstractmethod
    def load(self):
        """
        Загружает векторное хранилище с диска (если существует).

        Raises:
            NotImplementedError: Если метод не реализован в дочернем классе.
        """
        pass

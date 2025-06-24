from abc import ABC, abstractmethod
from utils.decorators import measure_time
from config.settings import logger
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingFactory:
    """
    Фабрика для создания объектов эмбеддингов различных типов.

    Предоставляет единый интерфейс для создания различных реализаций эмбеддингов.
    Позволяет легко добавлять новые типы эмбеддингов без изменения клиентского кода.

    Методы:
        create_embedding: Создает и возвращает объект эмбеддинга указанного типа.
    """

    @staticmethod
    @measure_time
    def create_embedding(embedding_type="huggingface", **kwargs):
        """
        Создает объект эмбеддинга заданного типа.

        Args:
            embedding_type (str): Тип создаваемого эмбеддинга. По умолчанию "huggingface".
            **kwargs: Дополнительные аргументы, передаваемые в конкретную реализацию эмбеддинга.

        Returns:
            Объект эмбеддинга указанного типа.

        Raises:
            ValueError: Если передан неизвестный тип эмбеддинга.
        """
        if embedding_type == "huggingface":
            logger.info(f"Проверка модели ...")
            return HuggingFaceEmbedding(**kwargs).get_embedding()
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")


class BaseEmbedding(ABC):
    """
    Абстрактный базовый класс для всех реализаций эмбеддингов.
    """

    @abstractmethod
    def get_embedding(self):
        """
        Абстрактный метод, который должен возвращать готовый к использованию объект эмбеддинга.

        Returns:
            Объект эмбеддинга (зависит от конкретной реализации).

        Raises:
            NotImplementedError: Если метод не реализован в дочернем классе.
        """
        pass


class HuggingFaceEmbedding(BaseEmbedding):
    """
    Конкретная реализация эмбеддинга для моделей Hugging Face.

    Создает и возвращает LangChain-совместимую обертку над SentenceTransformer.
    Поддерживает различные параметры модели и устройства (CPU/GPU).
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Инициализирует эмбеддинг Hugging Face.

        Args:
            model_name (str): Название или путь к модели Hugging Face.
            **kwargs: Дополнительные параметры:
                - model_kwargs (dict): Аргументы для модели, например {"device": "cuda"}
        """
        self.model_name = model_name
        self.kwargs = kwargs

    def get_embedding(self):
        """
        Создает и возвращает LangChain-совместимый объект HuggingFaceEmbeddings.

        Returns:
            HuggingFaceEmbeddings: Готовый к использованию объект для создания эмбеддингов.

        Процесс:
            1. Сначала создается SentenceTransformer с указанными параметрами
            2. Затем создается LangChain-совместимая обертка
            3. Возвращается готовый объект эмбеддинга
        """
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

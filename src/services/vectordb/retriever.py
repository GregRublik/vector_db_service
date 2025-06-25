from typing import List, Optional, Union, Sequence
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from config.settings import logger, settings

class Retriever:
    """
    Модуль для обработки и ранжирования результатов поиска из векторной БД.

    Поддерживает:
    - Ранжирование результатов с помощью cross-encoder модели (BAAI/bge-reranker-v2-m3)
    - Гибкую обработку результатов поиска перед передачей в LLM
    - Возможность расширения для других методов пост-обработки

    Атрибуты:
        rerank_model: Модель для переранжирования результатов
        rerank_tokenizer: Токенизатор для модели переранжирования
        device: Устройство для выполнения вычислений (cpu/cuda)
        model_max_length: Максимальная длина входных данных для модели
    """

    def __init__(
            self,
            rerank_model_name: Optional[str] = settings.rerank_model_name,
            device: Optional[str] = None,
            model_max_length: int = 512
    ):
        """
        Инициализация ретривера.

        Args:
            rerank_model_name: Название модели для переранжирования (None для отключения)
            device: Устройство для выполнения вычислений (None для автоопределения)
            model_max_length: Максимальная длина входных данных для модели
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_max_length = model_max_length
        self.rerank_model = None
        self.rerank_tokenizer = None

        if rerank_model_name:
            try:
                self._load_rerank_model(rerank_model_name)
            except Exception as e:
                logger.error(f"Failed to load rerank model: {e}")
                raise

    def _load_rerank_model(self, model_name: str):
        """Загружает модель и токенизатор для переранжирования."""
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.rerank_model.eval()
        self.rerank_model.to(self.device)
        logger.info(f"Rerank model loaded on {self.device}")

    def rerank_documents(
            self,
            query: str,
            documents: Sequence[Document],
            top_k: Optional[int] = None,
            batch_size: int = 32
    ) -> List[Document]:
        """
        Переранжирует документы с помощью cross-encoder модели.

        Args:
            query: Поисковый запрос
            documents: Последовательность документов для ранжирования
            top_k: Количество возвращаемых документов после ранжирования
            batch_size: Размер батча для обработки

        Returns:
            List[Document]: Переранжированные документы
        """
        if not documents:
            return []

        if not self.rerank_model or not self.rerank_tokenizer:
            logger.warning("Rerank model not initialized, returning original documents")
            return list(documents)[:top_k] if top_k else list(documents)

        try:
            # Разбиваем на батчи для обработки
            doc_list = list(documents)
            scores = []

            for i in range(0, len(doc_list), batch_size):
                batch = doc_list[i:i + batch_size]
                pairs = [[query, doc.page_content] for doc in batch]

                with torch.no_grad(), torch.inference_mode():
                    inputs = self.rerank_tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.model_max_length
                    ).to(self.device)

                    batch_scores = self.rerank_model(**inputs).logits.view(-1).float()
                    scores.extend(batch_scores.cpu().tolist())

            # Сортируем документы по убыванию релевантности
            scored_docs = list(zip(scores, doc_list))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            reranked_docs = [doc for _, doc in scored_docs]

            return reranked_docs[:top_k] if top_k else reranked_docs

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return list(documents)[:top_k] if top_k else list(documents)

    def process_search_results(
            self,
            query: str,
            search_results: Union[Sequence[Document], Sequence[str], Sequence[dict]],
            rerank: bool = True,
            top_k: Optional[int] = None,
            **kwargs
    ) -> List[Document]:
        """
        Обрабатывает результаты поиска с возможностью переранжирования.

        Args:
            query: Поисковый запрос
            search_results: Результаты поиска (могут быть Document, str или dict)
            rerank: Флаг для включения/отключения переранжирования
            top_k: Количество возвращаемых документов
            **kwargs: Дополнительные параметры для rerank_documents

        Returns:
            List[Document]: Обработанные документы
        """
        if not search_results:
            return []

        try:
            # Нормализация формата входных данных
            if isinstance(search_results[0], str):
                documents = [Document(page_content=text) for text in search_results]
            elif isinstance(search_results[0], dict):
                documents = [Document(**doc_dict) for doc_dict in search_results]
            else:
                documents = list(search_results)

            # Применяем переранжирование если требуется
            if rerank:
                documents = self.rerank_documents(query, documents, top_k, **kwargs)
            elif top_k:
                documents = documents[:top_k]

            return documents

        except Exception as e:
            logger.error(f"Error processing search results: {e}")
            return []


retriever = Retriever(device="cpu")

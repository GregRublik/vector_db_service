import os
from utils.docs_loader import load_markdown_documents

from manager import VectorDBManager
os.environ["TRUST_REMOTE_CODE"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Инициализация с FAISS or Chroma
manager = VectorDBManager(
    store_name="test_faiss",
    embedding_type="huggingface",
    vector_store_type="faiss",
    embedding_kwargs={
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "model_kwargs": {"device": "cpu"},  # or "cuda"
    }
)

# Загрузка документов
documents = load_markdown_documents("data.md")

# Добавление документов
manager.add_documents(documents)

# # Поиск
# results = manager.similarity_search("расскажи о компании", k=3)


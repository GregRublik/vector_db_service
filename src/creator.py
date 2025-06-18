from manager import VectorDBManager
from utils.docs_loader import load_markdown_documents

import os
os.environ["TRUST_REMOTE_CODE"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Инициализация с FAISS
manager = VectorDBManager(
    store_name="test_faiss",
    embedding_type="huggingface",
    vector_store_type="faiss",
    embedding_kwargs={
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_kwargs": {"device": "cuda"},
    }
)

# Инициализация с ChromaDB
# manager = VectorDBManager(
#     store_name="test_chroma",
#     embedding_type="huggingface",
#     vector_store_type="chroma",
#     embedding_kwargs={
#         # "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#         "model_name": "sentence-transformers/all-MiniLM-L6-v2",
#         "model_kwargs": {"device": "cuda"},
#     }
# )

# Загрузка документов
documents = load_markdown_documents("data.md")

# Добавление документов
# faiss_manager.add_documents(documents)
manager.add_documents(documents)

# Поиск
# faiss_results = faiss_manager.similarity_search("расскажи о компании", k=3)
chroma_results = manager.similarity_search("расскажи о компании", k=3)

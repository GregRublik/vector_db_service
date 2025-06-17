from manager import VectorDBManager
from utils.docs_loader import load_markdown_documents

import os
os.environ["TRUST_REMOTE_CODE"] = "true"

## Инициализация с FAISS
# faiss_manager = VectorDBManager(
#     store_name="test_faiss",
#     embedding_type="huggingface",
#     vector_store_type="faiss",
#     embedding_kwargs={
#         "model_name": "ai-sage/Giga-Embeddings-instruct",
#         "model_kwargs": {"device": "cpu"}
#     }
# )

# Инициализация с ChromaDB
chroma_manager = VectorDBManager(
    store_name="test_chroma",
    embedding_type="huggingface",
    vector_store_type="chroma",
    embedding_kwargs={
        # "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_kwargs": {"device": "cpu"},
        "trust_remote_code": True,
    }
)

# Загрузка документов
documents = load_markdown_documents("data.md")

# Добавление документов
# faiss_manager.add_documents(documents)
chroma_manager.add_documents(documents)

# Поиск
# faiss_results = faiss_manager.similarity_search("расскажи о компании", k=3)
chroma_results = chroma_manager.similarity_search("расскажи о компании", k=3)

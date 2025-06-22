from langchain_community.vectorstores import FAISS
from services.vectordb.vector_stores.base import VectorStoreInterface
from utils.decorators import measure_time


class FAISSStore(VectorStoreInterface):
    def __init__(self, embedding, store_path, store_name):
        self.embedding = embedding
        self.store_path = store_path
        self.store_name = store_name
        self.store = None

    @measure_time
    def add_documents(self, documents):
        if self.store is None:
            self.store = FAISS.from_documents(documents, self.embedding)
        else:
            self.store.add_documents(documents)

    @measure_time
    def similarity_search(self, query, **kwargs):
        return self.store.similarity_search(query, **kwargs)

    def save(self):
        self.store.save_local(self.store_path, self.store_name)

    def load(self):
        self.store = FAISS.load_local(
            self.store_path,
            self.store_name,
            self.embedding,
            allow_dangerous_deserialization=True
        )
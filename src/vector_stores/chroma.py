from langchain_community.vectorstores import Chroma
from vector_stores.base import VectorStoreInterface


class ChromaStore(VectorStoreInterface):
    def __init__(self, embedding, store_path, collection_name):
        self.embedding = embedding
        self.store_path = store_path
        self.collection_name = collection_name
        self.store = None

    def add_documents(self, documents):
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
        return self.store.similarity_search(query, **kwargs)

    def save(self):
        self.store.persist()

    def load(self):
        self.store = Chroma(
            persist_directory=self.store_path,
            embedding_function=self.embedding,
            collection_name=self.collection_name
        )
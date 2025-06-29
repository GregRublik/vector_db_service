from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from config import settings


def load_markdown_documents(file_path: str) -> List[Document]:
    text = settings.base_dir / "src" / "docs" / file_path

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "topic"),
            ("##", "subtopic"),
        ]
    )

    chunks = splitter.split_text(text.read_text(encoding='utf-8'))

    documents = []
    for chunk in chunks:
        doc = f"Тема: {chunk.metadata['topic']}\nОписание: {chunk.metadata.get('subtopic')}\nИнформация:{chunk.page_content}"
        documents.append(Document(doc.lower(), metadata=chunk.metadata))

    return documents

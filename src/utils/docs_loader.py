from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from config.settings import settings


def load_markdown_documents(file_path: str) -> List[Document]:
    text = settings.base_dir / "docs" / file_path

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "topic"),
            ("##", "subtopic"),
        ]
    )

    chunks = splitter.split_text(text.read_text(encoding='utf-8'))

    documents = []
    for chunk in chunks:
        doc = f"""
Тема: {chunk.metadata['topic']}
Описание: {chunk.metadata.get('subtopic')}
Информация:{chunk.page_content}"""
        chunk.metadata['document_name'] = file_path
        documents.append(Document(doc.lower(), metadata=chunk.metadata))

    return documents

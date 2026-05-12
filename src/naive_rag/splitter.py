from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from naive_rag.config import Settings


def split_documents(docs: list[Document]) -> list[Document]:
    settings = Settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)

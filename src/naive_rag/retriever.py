from typing import Optional

from langchain_core.retrievers import BaseRetriever

from naive_rag.config import Settings
from naive_rag.vectorstore import build_vectorstore


def build_retriever(collection_name: Optional[str] = None) -> BaseRetriever:
    settings = Settings()
    vs = build_vectorstore(collection_name=collection_name)
    return vs.as_retriever(search_kwargs={"k": settings.retriever_k})

from langchain_core.retrievers import BaseRetriever

from naive_rag.config import Settings
from naive_rag.vectorstore import build_vectorstore


def build_retriever() -> BaseRetriever:
    settings = Settings()
    vs = build_vectorstore()
    return vs.as_retriever(search_kwargs={"k": settings.retriever_k})

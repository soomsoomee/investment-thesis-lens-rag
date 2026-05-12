import pytest
from langchain_core.documents import Document

from naive_rag.retriever import build_retriever


@pytest.mark.integration
def test_retriever_returns_documents():
    retriever = build_retriever()
    results = retriever.invoke("매크로 지표")
    assert isinstance(results, list)
    assert len(results) <= 4
    assert all(isinstance(d, Document) for d in results)

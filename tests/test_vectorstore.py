import pytest

from naive_rag.vectorstore import build_vectorstore, ingest


@pytest.mark.integration
def test_build_vectorstore_returns_pgvector():
    vs = build_vectorstore()
    # PGVector exposes similarity_search method
    assert hasattr(vs, "similarity_search")
    assert hasattr(vs, "add_documents")


@pytest.mark.integration
def test_ingest_then_search_finds_at_least_one():
    ingest(limit=2)
    vs = build_vectorstore()
    results = vs.similarity_search("투자", k=1)
    assert len(results) >= 1

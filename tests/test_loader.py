import pytest
from langchain_core.documents import Document

from naive_rag.loader import load_transcripts


@pytest.mark.integration
def test_load_transcripts_returns_documents():
    docs = load_transcripts(limit=2)
    assert isinstance(docs, list)
    assert len(docs) >= 1
    assert all(isinstance(d, Document) for d in docs)
    assert all(d.page_content for d in docs)
    assert all("content_id" in d.metadata for d in docs)

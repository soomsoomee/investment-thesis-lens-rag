import pytest


@pytest.mark.integration
def test_ingest_populates_collection():
    from advanced_retrieval.ingest import ingest, build_vectorstore

    count = ingest()
    assert count > 0
    vs = build_vectorstore()
    hits = vs.similarity_search("엔비디아", k=1)
    assert len(hits) == 1

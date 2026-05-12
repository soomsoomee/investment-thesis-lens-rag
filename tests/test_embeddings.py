import pytest

from naive_rag.embeddings import build_embeddings


@pytest.mark.integration
def test_embeddings_returns_fixed_dim_vector():
    emb = build_embeddings()
    vec = emb.embed_query("투자 매크로 지표")
    assert isinstance(vec, list)
    assert len(vec) == 1024  # bge-m3 dim
    assert all(isinstance(x, float) for x in vec[:5])

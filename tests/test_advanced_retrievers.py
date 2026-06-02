import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


def test_kiwi_tokenize_korean():
    from advanced_retrieval.retrievers import kiwi_tokenize

    toks = kiwi_tokenize("엔비디아가 GTC에서 발표했다")
    assert isinstance(toks, list)
    assert all(isinstance(t, str) for t in toks)
    assert any("엔비디아" in t or t == "엔비디아" for t in toks)


class _FakeRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content=str(i)) for i in range(10)]


def test_topk_retriever_caps():
    from advanced_retrieval.retrievers import TopKRetriever

    r = TopKRetriever(base=_FakeRetriever(), k=4)
    out = r.invoke("q")
    assert len(out) == 4


@pytest.mark.integration
@pytest.mark.parametrize(
    "builder_name", ["build_dense", "build_bm25", "build_hybrid", "build_hybrid_rerank"]
)
def test_builders_return_capped_documents(builder_name):
    import advanced_retrieval.retrievers as R

    retriever = getattr(R, builder_name)()
    docs = retriever.invoke("엔비디아 GTC 발표")
    assert isinstance(docs, list)
    assert 0 < len(docs) <= 4
    assert all(isinstance(d, Document) for d in docs)

import pytest
from langchain_core.documents import Document


def _doc(cid, idx, total=5, text=None):
    return Document(
        page_content=text or f"{cid}-{idx}",
        metadata={"content_id": cid, "chunk_index": idx, "chunk_total": total},
    )


def test_dedupe_docs():
    from agentic_rag.retrieval import dedupe_docs

    docs = [_doc("a", 0), _doc("a", 0), _doc("a", 1)]
    out = dedupe_docs(docs)
    assert len(out) == 2


def test_multi_retrieve_unions_and_dedupes():
    from agentic_rag.retrieval import multi_retrieve

    class _R:
        def __init__(self, docs):
            self._docs = docs

        def invoke(self, q):
            return self._docs

    r = _R([_doc("a", 0), _doc("a", 1)])
    out = multi_retrieve(r, ["q1", "q2"])  # 같은 결과 2번 → dedupe
    assert len(out) == 2


@pytest.mark.integration
def test_expand_neighbors_adds_adjacent():
    from agentic_rag.retrieval import build_clean_retriever, expand_neighbors

    base = build_clean_retriever(k=1).invoke("엔비디아 GTC 발표")
    expanded = expand_neighbors(base)
    assert len(expanded) >= len(base)
    keys = {(d.metadata["content_id"], d.metadata["chunk_index"]) for d in expanded}
    assert len(keys) == len(expanded)  # dedupe 유지

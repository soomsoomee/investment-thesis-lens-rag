from langchain_core.documents import Document


def _doc(cid="a", idx=0):
    return Document(page_content="엔비디아", metadata={"content_id": cid, "chunk_index": idx, "chunk_total": 3,
                                                   "title": "t", "published_at": "2026-01-01"})


def test_generate_refuses_when_insufficient():
    from agentic_rag.nodes import generate_node
    from agentic_rag.state import REFUSAL, initial_state

    s = initial_state("q")
    s["grade_result"] = "insufficient"
    out = generate_node(s, llm=None)  # 거부 경로는 LLM 호출 안 함
    assert out["answer"] == REFUSAL


def test_retrieve_uses_subqueries_and_expands(monkeypatch):
    from agentic_rag import nodes
    from agentic_rag.state import initial_state

    class _R:
        def invoke(self, q):
            return [_doc(idx=0)]

    monkeypatch.setattr(nodes, "multi_retrieve", lambda r, qs: [_doc(idx=0)])
    monkeypatch.setattr(nodes, "expand_neighbors", lambda docs, vs=None: docs + [_doc(idx=1)])

    s = initial_state("q")
    s["sub_queries"] = ["sub1", "sub2"]
    out = nodes.retrieve_node(s, retriever=_R(), vs=None)
    assert len(out["documents"]) == 2
    assert "retrieve" in out["route_history"]

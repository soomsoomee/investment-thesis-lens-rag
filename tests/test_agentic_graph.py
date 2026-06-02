import pytest


@pytest.mark.integration
@pytest.mark.parametrize("with_reflect", [False, True])
def test_graph_answers(with_reflect):
    from agentic_rag.graph import build_graph
    from agentic_rag.state import initial_state

    graph = build_graph(with_reflect=with_reflect)
    final = graph.invoke(initial_state("GTC 워싱턴에서 엔비디아가 발표한 신규 사업 영역은?"))
    assert isinstance(final["answer"], str) and final["answer"]
    assert final["retry_count"] <= 2
    assert "retrieve" in final["route_history"]


@pytest.mark.integration
def test_graph_mermaid_renders():
    from agentic_rag.graph import build_graph

    m = build_graph(with_reflect=True).get_graph().draw_mermaid()
    assert "retrieve" in m and "grade_documents" in m

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


@pytest.mark.integration
def test_adaptive_graph_routes_complex_through_decompose():
    from agentic_rag.graph import build_graph
    from agentic_rag.state import initial_state

    graph = build_graph(adaptive=True)
    final = graph.invoke(initial_state("AI 인프라 투자 붐의 수혜주를 분야별로 정리하면?"))
    assert isinstance(final["answer"], str) and final["answer"]
    # 복합 질문이면 analyze→decompose를 거쳐 sub_queries가 채워진다
    assert final["complexity"] == "complex"
    assert len(final["sub_queries"]) >= 2
    assert "decompose" in final["route_history"]

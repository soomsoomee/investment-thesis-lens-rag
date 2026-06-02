import pytest


@pytest.mark.integration
def test_eval_graph_smoke():
    from agentic_rag.eval import eval_graph
    from agentic_rag.graph import build_graph
    from naive_rag.constants import WEEK4_EVAL_DATA

    df = eval_graph(build_graph(with_reflect=False), eval_data=WEEK4_EVAL_DATA[:2])
    assert len(df) == 2
    for col in ["faithfulness", "answer_relevancy", "latency_s", "llm_calls"]:
        assert any(col in c for c in df.columns)

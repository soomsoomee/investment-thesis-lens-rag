import pytest


@pytest.mark.integration
def test_run_eval_dense_smoke():
    from advanced_retrieval.eval import run_eval
    from advanced_retrieval.retrievers import build_dense
    from naive_rag.constants import WEEK4_EVAL_DATA

    result = run_eval(build_dense(), eval_data=WEEK4_EVAL_DATA[:2])
    df = result.to_pandas()
    assert len(df) == 2
    for col in ["faithfulness", "answer_relevancy"]:
        assert any(col in c for c in df.columns)

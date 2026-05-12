import pytest

from naive_rag.chain import build_chain


@pytest.mark.integration
def test_chain_answers_question():
    chain = build_chain()
    out = chain.invoke("최근 트랜스크립트에서 자주 인용되는 매크로 지표는?")
    assert isinstance(out, str)
    assert len(out) > 0

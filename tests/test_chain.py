import pytest
from langchain_core.documents import Document

from naive_rag.chain import PROMPT_TEMPLATE, _format_docs, build_chain


@pytest.mark.integration
def test_chain_answers_question():
    chain = build_chain()
    out = chain.invoke("최근 트랜스크립트에서 자주 인용되는 매크로 지표는?")
    assert isinstance(out, str)
    assert len(out) > 0


def test_format_docs_includes_published_at_in_header():
    docs = [
        Document(
            page_content="본문 내용",
            metadata={"channel": "소수몽키", "title": "어떤 영상", "published_at": "2026-04-28"},
        )
    ]
    formatted = _format_docs(docs)
    assert "소수몽키" in formatted
    assert "어떤 영상" in formatted
    assert "2026-04-28" in formatted
    assert "본문 내용" in formatted


def test_prompt_template_instructs_to_cite_sources():
    assert "[출처]" in PROMPT_TEMPLATE

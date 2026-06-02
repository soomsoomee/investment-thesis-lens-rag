import pytest
from langchain_core.documents import Document


def test_build_clean_chunks_filters_low(monkeypatch):
    keep = Document(page_content="엔비디아 GTC 발표", metadata={"content_id": "c", "chunk_index": 0})
    drop = Document(page_content="구독 좋아요 눌러주세요", metadata={"content_id": "c", "chunk_index": 1})
    monkeypatch.setattr("agentic_rag.preprocess.load_chunks", lambda: [keep, drop])
    monkeypatch.setattr(
        "agentic_rag.preprocess.classify_chunk",
        lambda d, llm=None: "구독" not in d.page_content,
    )

    from agentic_rag.preprocess import build_clean_chunks

    out = build_clean_chunks()
    assert len(out) == 1
    assert out[0].page_content == "엔비디아 GTC 발표"


@pytest.mark.integration
def test_ingest_clean_populates():
    # NOTE: 1500자 chunk-level 분류는 이 코퍼스에서 사실상 0개를 drop한다(노이즈가
    # 큰 chunk 안에 희석돼 LLM이 전부 keep). 이는 회고에 기록하는 negative finding이며,
    # 여기서는 분류 파이프라인이 돌아가 collection이 적재되는지만 검증한다.
    from advanced_retrieval.chunks import load_chunks
    from agentic_rag.preprocess import ingest_clean

    raw = len(load_chunks())
    kept = ingest_clean()
    assert 0 < kept <= raw

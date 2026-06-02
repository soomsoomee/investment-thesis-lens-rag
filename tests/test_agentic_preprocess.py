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
def test_ingest_clean_drops_some():
    from advanced_retrieval.chunks import load_chunks
    from agentic_rag.preprocess import build_clean_chunks, ingest_clean

    raw = len(load_chunks())
    kept = ingest_clean()
    assert 0 < kept <= raw  # 일부는 노이즈로 drop

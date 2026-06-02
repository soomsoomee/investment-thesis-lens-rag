from langchain_core.documents import Document


def test_load_chunks_splits_and_enriches(monkeypatch):
    long_text = ("투자 이야기. " * 400).strip()  # ~1500자 훌쩍 넘김
    fake_doc = Document(
        page_content=long_text,
        metadata={"content_id": "cid-1", "title": "t", "channel": "c", "published_at": "2026-01-01"},
    )
    monkeypatch.setattr(
        "advanced_retrieval.chunks.load_transcripts",
        lambda content_ids=None: [fake_doc],
    )

    from advanced_retrieval.chunks import load_chunks

    chunks = load_chunks()
    assert len(chunks) >= 2  # 1500 size로 쪼개지면 2개 이상
    assert all("chunk_index" in c.metadata for c in chunks)
    assert all("chunk_total" in c.metadata for c in chunks)
    assert all("chunk_token_count" in c.metadata for c in chunks)
    assert chunks[0].metadata["content_id"] == "cid-1"

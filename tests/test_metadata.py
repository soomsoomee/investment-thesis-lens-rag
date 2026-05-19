from langchain_core.documents import Document

from naive_rag.metadata import enrich_chunks


def _doc(content: str, content_id: str) -> Document:
    return Document(page_content=content, metadata={"content_id": content_id})


def test_enrich_chunks_adds_index_total_and_token_count():
    chunks = [
        _doc("첫 번째 청크입니다.", "A"),
        _doc("두 번째 청크입니다.", "A"),
        _doc("문서 B의 유일한 청크.", "B"),
    ]

    enriched = enrich_chunks(chunks)

    # doc A
    assert enriched[0].metadata["chunk_index"] == 0
    assert enriched[0].metadata["chunk_total"] == 2
    assert enriched[0].metadata["chunk_token_count"] > 0
    assert enriched[1].metadata["chunk_index"] == 1
    assert enriched[1].metadata["chunk_total"] == 2

    # doc B
    assert enriched[2].metadata["chunk_index"] == 0
    assert enriched[2].metadata["chunk_total"] == 1

    # 원본 metadata 보존
    assert enriched[0].metadata["content_id"] == "A"
    assert enriched[2].metadata["content_id"] == "B"


def test_enrich_chunks_preserves_order_within_each_doc():
    # 같은 content_id의 chunk 순서가 입력 순서와 동일해야 함
    chunks = [_doc(f"청크 {i}", "X") for i in range(5)]
    enriched = enrich_chunks(chunks)
    for i, c in enumerate(enriched):
        assert c.metadata["chunk_index"] == i
        assert c.metadata["chunk_total"] == 5


def test_enrich_chunks_returns_new_list_does_not_mutate_input():
    original = [_doc("청크", "A")]
    enriched = enrich_chunks(original)
    assert "chunk_index" not in original[0].metadata
    assert enriched is not original

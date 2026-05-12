from langchain_core.documents import Document

from naive_rag.splitter import split_documents


def test_split_documents_chunks_long_text():
    long_text = "한 문장. " * 500  # well above chunk_size
    docs = [Document(page_content=long_text, metadata={"content_id": "x"})]

    chunks = split_documents(docs)

    assert len(chunks) > 1
    for c in chunks:
        assert c.page_content
        assert c.metadata["content_id"] == "x"


def test_split_documents_preserves_short_doc():
    docs = [Document(page_content="짧은 글.", metadata={"content_id": "y"})]
    chunks = split_documents(docs)
    assert len(chunks) == 1
    assert chunks[0].page_content == "짧은 글."

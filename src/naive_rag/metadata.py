"""Chunk metadata 후처리.

splitter가 만든 chunk Document에 chunk_index/chunk_total/chunk_token_count를 채워넣는다.
content_id 기준으로 그룹핑한 뒤 입력 순서대로 index 부여.
"""
from collections import defaultdict

import tiktoken
from langchain_core.documents import Document

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def enrich_chunks(chunks: list[Document]) -> list[Document]:
    """Add chunk_index / chunk_total / chunk_token_count to each chunk's metadata.

    원본 list와 Document를 mutate하지 않고 새 list/Document를 반환한다.
    Index는 content_id별로 0부터 시작하며 입력 순서를 따른다.
    """
    totals: dict[str, int] = defaultdict(int)
    for c in chunks:
        totals[c.metadata.get("content_id", "")] += 1

    counters: dict[str, int] = defaultdict(int)
    enriched: list[Document] = []
    for c in chunks:
        cid = c.metadata.get("content_id", "")
        new_meta = dict(c.metadata)
        new_meta["chunk_index"] = counters[cid]
        new_meta["chunk_total"] = totals[cid]
        new_meta["chunk_token_count"] = _count_tokens(c.page_content)
        counters[cid] += 1
        enriched.append(Document(page_content=c.page_content, metadata=new_meta))
    return enriched

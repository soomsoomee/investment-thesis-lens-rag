"""clean collection 위 Dense + neighbor expansion(±1) + multi-query union.

neighbor expansion은 검색된 chunk의 (content_id, chunk_index)만으로 ±1 이웃을
metadata filter로 가져온다 — 카탈로그 전체를 훑지 않아 컨텐츠가 늘어도 확장 안전.
"""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from agentic_rag.config import AgenticSettings
from agentic_rag.preprocess import build_clean_vectorstore


def _key(d: Document) -> tuple:
    return (d.metadata.get("content_id"), d.metadata.get("chunk_index"))


def dedupe_docs(docs: list[Document]) -> list[Document]:
    seen, out = set(), []
    for d in docs:
        k = _key(d)
        if k not in seen:
            seen.add(k)
            out.append(d)
    return out


def build_clean_retriever(k: int | None = None) -> BaseRetriever:
    s = AgenticSettings()
    vs = build_clean_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k or s.retriever_k})


def multi_retrieve(retriever: BaseRetriever, queries: list[str]) -> list[Document]:
    docs: list[Document] = []
    for q in queries:
        docs.extend(retriever.invoke(q))
    return dedupe_docs(docs)


def expand_neighbors(docs: list[Document], vs=None, window: int | None = None) -> list[Document]:
    """각 chunk에 같은 content_id의 chunk_index ±window 이웃을 붙인다."""
    s = AgenticSettings()
    window = window or s.neighbor_window
    vs = vs or build_clean_vectorstore()
    out = list(docs)
    for d in docs:
        cid = d.metadata.get("content_id")
        idx = d.metadata.get("chunk_index")
        total = d.metadata.get("chunk_total", 10**9)
        if cid is None or idx is None:
            continue
        wanted = [i for i in range(idx - window, idx + window + 1) if i != idx and 0 <= i < total]
        if not wanted:
            continue
        neighbors = vs.similarity_search(
            d.page_content,
            k=len(wanted),
            filter={"content_id": {"$eq": cid}, "chunk_index": {"$in": wanted}},
        )
        out.extend(neighbors)
    return dedupe_docs(out)

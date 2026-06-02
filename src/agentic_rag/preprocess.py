"""도입부/꼬리 노이즈를 ingest 단계에서 제거해 clean collection을 만든다.

LLM utility 분류기가 chunk별 keep/drop을 판정. per-chunk O(n) 1회라 컨텐츠가
늘어도 동일하게 작동(확장 안전).
"""
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector
from pydantic import BaseModel

from advanced_retrieval.chunks import load_chunks
from agentic_rag.config import AgenticSettings
from naive_rag.config import Settings
from naive_rag.embeddings import build_embeddings

CLASSIFY_PROMPT = """다음은 투자 유튜브 트랜스크립트의 한 조각이다.
이 조각이 투자 분석/정보(종목, 시장, 전망, 수치, 기업 동향 등)를 담고 있으면 keep=true,
단순 인사·주간 예고·구독/좋아요/카페/댓글/특강 안내 같은 곁가지면 keep=false로 판정하라.

[조각]
{chunk}
"""


class _Utility(BaseModel):
    keep: bool


def _llm() -> ChatOpenAI:
    s = Settings()
    return ChatOpenAI(model=s.llm_model, api_key=s.openai_api_key, temperature=0)


def classify_chunk(doc: Document, llm: ChatOpenAI | None = None) -> bool:
    llm = llm or _llm()
    decision = llm.with_structured_output(_Utility).invoke(
        CLASSIFY_PROMPT.format(chunk=doc.page_content)
    )
    return decision.keep


def build_clean_chunks() -> list[Document]:
    llm = _llm()
    return [c for c in load_chunks() if classify_chunk(c, llm)]


def build_clean_vectorstore() -> PGVector:
    s = AgenticSettings()
    base = Settings()
    return PGVector(
        embeddings=build_embeddings(),
        collection_name=s.clean_collection_name,
        connection=base.main_database_url,
        use_jsonb=True,
    )


def _is_populated(vs: PGVector) -> bool:
    try:
        return len(vs.similarity_search("투자", k=1)) > 0
    except Exception:
        return False


def ingest_clean(force: bool = False) -> int:
    vs = build_clean_vectorstore()
    chunks = build_clean_chunks()
    if force or not _is_populated(vs):
        vs.add_documents(chunks)
    return len(chunks)

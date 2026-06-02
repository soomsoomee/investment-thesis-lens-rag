"""study_advanced_retrieval collection에 B-large 청크 적재.

이미 채워져 있으면(force=False) 재적재하지 않아 중복을 막는다.
"""
from langchain_postgres import PGVector

from advanced_retrieval.chunks import load_chunks
from advanced_retrieval.config import AdvancedSettings
from naive_rag.config import Settings
from naive_rag.embeddings import build_embeddings


def build_vectorstore() -> PGVector:
    s = AdvancedSettings()
    base = Settings()
    return PGVector(
        embeddings=build_embeddings(),
        collection_name=s.collection_name,
        connection=base.main_database_url,
        use_jsonb=True,
    )


def _is_populated(vs: PGVector) -> bool:
    try:
        return len(vs.similarity_search("투자", k=1)) > 0
    except Exception:
        return False


def ingest(force: bool = False) -> int:
    vs = build_vectorstore()
    chunks = load_chunks()
    if force or not _is_populated(vs):
        vs.add_documents(chunks)
    return len(chunks)

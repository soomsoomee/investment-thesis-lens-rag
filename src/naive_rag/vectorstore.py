import sys
from typing import Optional

from langchain_postgres import PGVector

from naive_rag.config import Settings
from naive_rag.embeddings import build_embeddings
from naive_rag.loader import load_transcripts
from naive_rag.splitter import split_documents


def build_vectorstore() -> PGVector:
    settings = Settings()
    return PGVector(
        embeddings=build_embeddings(),
        collection_name=settings.collection_name,
        connection=settings.main_database_url,
        use_jsonb=True,
    )


def ingest(limit: Optional[int] = None) -> int:
    docs = load_transcripts(limit=limit)
    chunks = split_documents(docs)
    vs = build_vectorstore()
    vs.add_documents(chunks)
    return len(chunks)


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "ingest":
        limit = int(argv[2]) if len(argv) >= 3 else None
        count = ingest(limit=limit)
        print(f"Ingested {count} chunks into collection 'study_naive_rag'.")
        return 0
    print("Usage: python -m naive_rag.vectorstore ingest [limit]", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

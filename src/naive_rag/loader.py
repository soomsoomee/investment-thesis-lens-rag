from typing import Optional

from langchain_core.documents import Document
from sqlalchemy import create_engine, text

from naive_rag.config import Settings


def load_transcripts(limit: Optional[int] = None) -> list[Document]:
    settings = Settings()
    engine = create_engine(settings.main_database_url)
    sql = """
        SELECT
            c.id AS id,
            c.transcript AS transcript,
            c.title AS title,
            p.name AS channel_name,
            c.published_at AS published_at
        FROM contents c
        LEFT JOIN personas p ON p.id = c.persona_id
        WHERE c.transcript IS NOT NULL AND c.transcript <> ''
        ORDER BY c.published_at DESC NULLS LAST
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    docs: list[Document] = []
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    for row in rows:
        metadata = {
            "content_id": str(row["id"]),
            "title": row["title"] or "",
            "channel": row["channel_name"] or "",
            "published_at": str(row["published_at"]) if row["published_at"] else "",
        }
        docs.append(Document(page_content=row["transcript"], metadata=metadata))
    return docs

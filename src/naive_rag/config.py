import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    value = os.environ.get(key)
    if value is None:
        raise KeyError(f"Required env var not set: {key}")
    return value


@dataclass(frozen=True)
class Settings:
    main_database_url: str = field(default_factory=lambda: _require("MAIN_DATABASE_URL"))
    openai_api_key: str = field(default_factory=lambda: _require("OPENAI_API_KEY"))
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "gpt-4o-mini"
    collection_name: str = "study_naive_rag"
    chunk_size: int = 600
    chunk_overlap: int = 100
    retriever_k: int = 4

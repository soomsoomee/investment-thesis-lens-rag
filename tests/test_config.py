import os
import pytest


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("MAIN_DATABASE_URL", "postgresql+psycopg://u:p@h:5432/d")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from naive_rag.config import Settings

    s = Settings()
    assert s.main_database_url == "postgresql+psycopg://u:p@h:5432/d"
    assert s.openai_api_key == "sk-test"
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.llm_model == "gpt-4o-mini"
    assert s.collection_name == "study_naive_rag"
    assert s.chunk_size == 600
    assert s.chunk_overlap == 100
    assert s.retriever_k == 4


def test_settings_missing_env_raises(monkeypatch):
    monkeypatch.delenv("MAIN_DATABASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from naive_rag.config import Settings

    with pytest.raises(KeyError):
        Settings()

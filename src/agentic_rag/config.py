from dataclasses import dataclass


@dataclass(frozen=True)
class AgenticSettings:
    """6주차 Agentic RAG 전용 knob. 공유 설정은 naive_rag.config.Settings 재사용."""

    clean_collection_name: str = "study_agentic_clean"
    retriever_k: int = 4
    neighbor_window: int = 1
    max_retries: int = 2

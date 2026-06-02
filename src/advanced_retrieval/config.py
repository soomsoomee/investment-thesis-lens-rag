from dataclasses import dataclass


@dataclass(frozen=True)
class AdvancedSettings:
    """5주차 retrieval ablation 전용 knob. DB/임베딩/LLM 등 공유 설정은
    naive_rag.config.Settings를 그대로 재사용한다."""

    collection_name: str = "study_advanced_retrieval"
    chunk_size: int = 1500
    chunk_overlap: int = 300
    separators: tuple = ("\n\n", "\n", ". ", " ", "")
    retriever_k: int = 4          # 최종 LLM 전달 chunk 수 (4 strategy 공통)
    hybrid_candidate_k: int = 8   # Hybrid에서 각 하위 retriever 후보 수
    rerank_candidate_k: int = 12  # rerank 전 hybrid 후보 수
    rerank_top_n: int = 4         # rerank 후 최종 수
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    ensemble_weights: tuple = (0.5, 0.5)

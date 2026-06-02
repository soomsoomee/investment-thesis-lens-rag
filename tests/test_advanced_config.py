def test_advanced_settings_defaults():
    from advanced_retrieval.config import AdvancedSettings

    s = AdvancedSettings()
    assert s.collection_name == "study_advanced_retrieval"
    assert s.chunk_size == 1500
    assert s.chunk_overlap == 300
    assert s.separators == ("\n\n", "\n", ". ", " ", "")
    assert s.retriever_k == 4
    assert s.hybrid_candidate_k == 8
    assert s.rerank_candidate_k == 12
    assert s.rerank_top_n == 4
    assert s.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert s.ensemble_weights == (0.5, 0.5)

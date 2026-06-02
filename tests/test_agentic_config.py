def test_agentic_settings_defaults():
    from agentic_rag.config import AgenticSettings

    s = AgenticSettings()
    assert s.clean_collection_name == "study_agentic_clean"
    assert s.retriever_k == 4
    assert s.neighbor_window == 1
    assert s.max_retries == 2

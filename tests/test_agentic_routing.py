def _state(**kw):
    from agentic_rag.state import initial_state

    s = initial_state("q")
    s.update(kw)
    return s


def test_route_enough_goes_generate():
    from agentic_rag.state import route_after_grade

    assert route_after_grade(_state(grade_result="enough")) == "generate"


def test_route_insufficient_retries():
    from agentic_rag.state import route_after_grade

    assert route_after_grade(_state(grade_result="insufficient", retry_count=0)) == "rewrite_query"


def test_route_insufficient_exhausted_goes_generate():
    from agentic_rag.state import route_after_grade

    assert route_after_grade(_state(grade_result="insufficient", retry_count=2)) == "generate"


def test_initial_state_shape():
    from agentic_rag.state import initial_state

    s = initial_state("질문")
    assert s["question"] == "질문"
    assert s["retry_count"] == 0
    assert s["sub_queries"] == []
    assert s["documents"] == []

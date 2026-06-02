from typing import Any

from langchain_core.documents import Document
from langgraph.graph import END
from typing_extensions import TypedDict

from agentic_rag.config import AgenticSettings

MAX_RETRIES = AgenticSettings().max_retries
REFUSAL = "제공된 문서에서는 해당 질문에 대한 근거를 확인할 수 없습니다."


class GraphState(TypedDict):
    question: str
    sub_queries: list[str]
    complexity: str
    documents: list[Document]
    answer: str
    grade_result: str
    retry_count: int
    route_history: list[str]
    latency: float


def initial_state(question: str) -> GraphState:
    return {
        "question": question,
        "sub_queries": [],
        "complexity": "",
        "documents": [],
        "answer": "",
        "grade_result": "",
        "retry_count": 0,
        "route_history": [],
        "latency": 0.0,
    }


def route_after_analyze(state: GraphState) -> str:
    return "decompose" if state["complexity"] == "complex" else "retrieve"


def route_after_grade(state: GraphState) -> str:
    if state["grade_result"] == "enough":
        return "generate"
    if state["retry_count"] >= MAX_RETRIES:
        return "generate"  # 재시도 소진 → generate가 답변 불가 처리
    return "rewrite_query"


def route_after_reflect(state: GraphState) -> Any:
    if state["grade_result"] == "insufficient" and state["retry_count"] < MAX_RETRIES:
        return "rewrite_query"
    return END

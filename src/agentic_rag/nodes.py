"""LangGraph 노드 함수 (의존성 주입형). graph.py가 retriever/vs/llm을 바인딩한다."""
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel

from agentic_rag.retrieval import expand_neighbors, multi_retrieve
from agentic_rag.state import REFUSAL, GraphState
from naive_rag.chain import PROMPT_TEMPLATE, _format_docs

GRADE_PROMPT = """질문에 답하는 데 관련 있는 조각의 번호만 고르라.
조각에는 제목/날짜가 함께 있다. 주제나 시점이 질문과 맞지 않으면 제외하라.

[질문]
{question}

[조각들]
{docs}
"""

REWRITE_PROMPT = """아래 질문이 너무 광범위하거나 검색에 부적합하다.
검색에 적합한 2~3개의 더 구체적인 한국어 sub-query로 분해하라(도메인 용어 포함).

[질문]
{question}
"""

REFLECT_PROMPT = """답변이 주어진 컨텍스트에 근거하는지 판정하라.
컨텍스트에 없는 내용을 지어냈으면 grounded=false.

[질문]
{question}

[컨텍스트]
{context}

[답변]
{answer}
"""

ANALYZE_PROMPT = """질문을 분류하라.
- simple: 단일 사실/한 종목·이벤트로 답할 수 있는 질문.
- complex: 여러 하위주제·여러 측면·'분야별/종합/정리'를 요구하는 질문.

[질문]
{question}
"""

DECOMPOSE_PROMPT = """아래 복합 질문을 검색에 적합한 2~4개의 구체적 한국어 sub-query로 분해하라.
하위주제별로 나누고, 도메인 용어를 포함하라.

[질문]
{question}
"""


class _GradeDecision(BaseModel):
    relevant_indices: list[int]


class _RewriteDecision(BaseModel):
    sub_queries: list[str]


class _ReflectDecision(BaseModel):
    grounded: bool


class _AnalyzeDecision(BaseModel):
    complexity: Literal["simple", "complex"]


class _DecomposeDecision(BaseModel):
    sub_queries: list[str]


def _listing(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (제목: {d.metadata.get('title','?')} / 날짜: {d.metadata.get('published_at','?')})\n"
        f"{d.page_content[:300]}"
        for i, d in enumerate(docs)
    )


def analyze_query_node(state: GraphState, llm) -> dict:
    decision = llm.with_structured_output(_AnalyzeDecision).invoke(
        ANALYZE_PROMPT.format(question=state["question"])
    )
    return {"complexity": decision.complexity,
            "route_history": state["route_history"] + [f"analyze({decision.complexity})"]}


def decompose_node(state: GraphState, llm) -> dict:
    decision = llm.with_structured_output(_DecomposeDecision).invoke(
        DECOMPOSE_PROMPT.format(question=state["question"])
    )
    return {"sub_queries": decision.sub_queries or [state["question"]],
            "route_history": state["route_history"] + ["decompose"]}


def retrieve_node(state: GraphState, retriever, vs) -> dict:
    queries = state["sub_queries"] or [state["question"]]
    docs = multi_retrieve(retriever, queries)
    docs = expand_neighbors(docs, vs=vs)
    return {"documents": docs, "route_history": state["route_history"] + ["retrieve"]}


def grade_node(state: GraphState, llm) -> dict:
    docs = state["documents"]
    if not docs:
        return {"documents": [], "grade_result": "insufficient",
                "route_history": state["route_history"] + ["grade"]}
    decision = llm.with_structured_output(_GradeDecision).invoke(
        GRADE_PROMPT.format(question=state["question"], docs=_listing(docs))
    )
    relevant = [docs[i] for i in decision.relevant_indices if 0 <= i < len(docs)]
    return {
        "documents": relevant,
        "grade_result": "enough" if relevant else "insufficient",
        "route_history": state["route_history"] + ["grade"],
    }


def rewrite_node(state: GraphState, llm) -> dict:
    decision = llm.with_structured_output(_RewriteDecision).invoke(
        REWRITE_PROMPT.format(question=state["question"])
    )
    return {
        "sub_queries": decision.sub_queries or [state["question"]],
        "retry_count": state["retry_count"] + 1,
        "route_history": state["route_history"] + ["rewrite"],
    }


def generate_node(state: GraphState, llm) -> dict:
    if state["grade_result"] != "enough" or not state["documents"]:
        return {"answer": REFUSAL, "route_history": state["route_history"] + ["generate(refuse)"]}
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {"context": _format_docs(state["documents"]), "question": state["question"]}
    )
    return {"answer": answer, "route_history": state["route_history"] + ["generate"]}


def reflect_node(state: GraphState, llm) -> dict:
    decision = llm.with_structured_output(_ReflectDecision).invoke(
        REFLECT_PROMPT.format(
            question=state["question"],
            context=_format_docs(state["documents"]),
            answer=state["answer"],
        )
    )
    if decision.grounded:
        return {"route_history": state["route_history"] + ["reflect(ok)"]}
    return {"grade_result": "insufficient",
            "route_history": state["route_history"] + ["reflect(fail)"]}

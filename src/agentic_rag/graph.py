"""retrieve→grade→rewrite→generate (+retry, +reflect) 그래프 조립."""
from functools import partial

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agentic_rag import nodes
from agentic_rag.preprocess import build_clean_vectorstore
from agentic_rag.retrieval import build_clean_retriever
from agentic_rag.state import (
    GraphState,
    route_after_analyze,
    route_after_grade,
    route_after_reflect,
)
from naive_rag.config import Settings


def _llm() -> ChatOpenAI:
    s = Settings()
    return ChatOpenAI(model=s.llm_model, api_key=s.openai_api_key, temperature=0)


def build_graph(with_reflect: bool = False, adaptive: bool = False):
    retriever = build_clean_retriever()
    vs = build_clean_vectorstore()
    llm = _llm()

    g = StateGraph(GraphState)
    g.add_node("retrieve", partial(nodes.retrieve_node, retriever=retriever, vs=vs))
    g.add_node("grade_documents", partial(nodes.grade_node, llm=llm))
    g.add_node("rewrite_query", partial(nodes.rewrite_node, llm=llm))
    g.add_node("generate", partial(nodes.generate_node, llm=llm))

    if adaptive:
        # 진입부 router: 단답이면 바로 retrieve, 복합이면 multi-query 분해 후 retrieve
        g.add_node("analyze_query", partial(nodes.analyze_query_node, llm=llm))
        g.add_node("decompose", partial(nodes.decompose_node, llm=llm))
        g.add_edge(START, "analyze_query")
        g.add_conditional_edges(
            "analyze_query",
            route_after_analyze,
            {"decompose": "decompose", "retrieve": "retrieve"},
        )
        g.add_edge("decompose", "retrieve")
    else:
        g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    g.add_edge("rewrite_query", "retrieve")

    if with_reflect:
        g.add_node("reflect", partial(nodes.reflect_node, llm=llm))
        g.add_edge("generate", "reflect")
        g.add_conditional_edges(
            "reflect", route_after_reflect, {"rewrite_query": "rewrite_query", END: END}
        )
    else:
        g.add_edge("generate", END)

    return g.compile()

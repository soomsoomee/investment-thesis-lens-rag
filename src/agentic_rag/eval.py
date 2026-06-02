"""Agentic 그래프 / baseline 체인을 RAGAS + 운영지표(latency, LLM 호출수)로 평가."""
import time
from typing import Sequence

import pandas as pd
from datasets import Dataset
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from agentic_rag.state import initial_state
from naive_rag.config import Settings
from naive_rag.constants import WEEK4_EVAL_DATA
from naive_rag.embeddings import build_embeddings
from naive_rag.eval import _default_metrics


def _ragas(dataset: Dataset) -> pd.DataFrame:
    s = Settings()
    result = evaluate(
        dataset,
        metrics=_default_metrics(),
        llm=LangchainLLMWrapper(ChatOpenAI(model=s.llm_model, api_key=s.openai_api_key, temperature=0)),
        embeddings=LangchainEmbeddingsWrapper(build_embeddings()),
    )
    return result.to_pandas()


def eval_graph(graph, eval_data: Sequence[dict] = WEEK4_EVAL_DATA) -> pd.DataFrame:
    rows, ops = [], []
    for d in eval_data:
        t0 = time.perf_counter()
        with get_openai_callback() as cb:
            final = graph.invoke(initial_state(d["question"]))
        ops.append({"latency_s": time.perf_counter() - t0, "llm_calls": cb.successful_requests,
                    "route_history": " > ".join(final["route_history"])})
        rows.append({
            "user_input": d["question"],
            "reference": d["reference"],
            "response": final["answer"],
            "retrieved_contexts": [c.page_content for c in final["documents"]] or ["(없음)"],
        })
    df = _ragas(Dataset.from_list(rows))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(ops)], axis=1)


def eval_chain(retriever, eval_data: Sequence[dict] = WEEK4_EVAL_DATA) -> pd.DataFrame:
    from advanced_retrieval.eval import build_chain_for

    chain = build_chain_for(retriever)
    rows, ops = [], []
    for d in eval_data:
        t0 = time.perf_counter()
        with get_openai_callback() as cb:
            answer = chain.invoke(d["question"])
            contexts = [c.page_content for c in retriever.invoke(d["question"])]
        ops.append({"latency_s": time.perf_counter() - t0, "llm_calls": cb.successful_requests})
        rows.append({"user_input": d["question"], "reference": d["reference"],
                     "response": answer, "retrieved_contexts": contexts})
    df = _ragas(Dataset.from_list(rows))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(ops)], axis=1)

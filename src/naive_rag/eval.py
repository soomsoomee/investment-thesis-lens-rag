"""4주차 chunking 비교를 위한 eval 함수.

3주차 노트북 cell 20의 inline 코드를 함수화. collection_name을 받아
원하는 vector store를 retrieve 대상으로 삼고, ragas 4 metric으로 평가.
"""
from typing import Optional, Sequence

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from naive_rag.chain import build_chain
from naive_rag.config import Settings
from naive_rag.constants import WEEK4_EVAL_DATA
from naive_rag.embeddings import build_embeddings
from naive_rag.retriever import build_retriever


def _default_metrics():
    return [
        Faithfulness(),
        AnswerRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]


def build_eval_dataset(
    eval_data: Sequence[dict],
    collection_name: Optional[str] = None,
) -> Dataset:
    """질문 10개에 대해 retrieve + chain 호출 결과를 Dataset으로 묶음."""
    retriever = build_retriever(collection_name=collection_name)
    chain = build_chain(collection_name=collection_name)
    rows = [
        {
            "user_input": d["question"],
            "reference": d["reference"],
            "response": chain.invoke(d["question"]),
            "retrieved_contexts": [c.page_content for c in retriever.invoke(d["question"])],
        }
        for d in eval_data
    ]
    return Dataset.from_list(rows)


def run_eval(
    collection_name: Optional[str] = None,
    eval_data: Optional[Sequence[dict]] = None,
):
    """전체 평가 파이프라인. 결과 객체는 .to_pandas()로 DataFrame 변환 가능."""
    settings = Settings()
    dataset = build_eval_dataset(
        eval_data=eval_data or WEEK4_EVAL_DATA,
        collection_name=collection_name,
    )
    return evaluate(
        dataset,
        metrics=_default_metrics(),
        llm=LangchainLLMWrapper(
            ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.openai_api_key,
                temperature=0,
            )
        ),
        embeddings=LangchainEmbeddingsWrapper(build_embeddings()),
    )

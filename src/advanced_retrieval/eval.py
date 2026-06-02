"""retriever 주입형 RAGAS 평가. naive_rag.eval은 collection만 받고 내부에서
dense retriever를 만들어 5주차 비교에 부적합 → retriever 자체를 주입하도록 일반화.
프롬프트/포맷/metric은 naive_rag 것을 그대로 재사용한다."""
from typing import Sequence

from datasets import Dataset
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from naive_rag.chain import PROMPT_TEMPLATE, _format_docs
from naive_rag.config import Settings
from naive_rag.constants import WEEK4_EVAL_DATA
from naive_rag.embeddings import build_embeddings
from naive_rag.eval import _default_metrics


def build_chain_for(retriever: BaseRetriever) -> Runnable:
    s = Settings()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model=s.llm_model, api_key=s.openai_api_key, temperature=0)
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def build_eval_dataset(
    retriever: BaseRetriever, eval_data: Sequence[dict] = WEEK4_EVAL_DATA
) -> Dataset:
    chain = build_chain_for(retriever)
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


def run_eval(retriever: BaseRetriever, eval_data: Sequence[dict] = WEEK4_EVAL_DATA):
    s = Settings()
    dataset = build_eval_dataset(retriever, eval_data)
    return evaluate(
        dataset,
        metrics=_default_metrics(),
        llm=LangchainLLMWrapper(
            ChatOpenAI(model=s.llm_model, api_key=s.openai_api_key, temperature=0)
        ),
        embeddings=LangchainEmbeddingsWrapper(build_embeddings()),
    )

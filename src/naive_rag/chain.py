from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

from naive_rag.config import Settings
from naive_rag.retriever import build_retriever

PROMPT_TEMPLATE = """당신은 투자 분석 트랜스크립트를 참고해 한국어로 답하는 도우미입니다.
아래 컨텍스트만 사용해 질문에 답하세요. 컨텍스트에 답이 없으면 "주어진 자료로는 알 수 없습니다."라고 답하세요.

[컨텍스트]
{context}

[질문]
{question}

[답변]
"""


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"({d.metadata.get('channel', '?')} / {d.metadata.get('title', '?')})\n{d.page_content}"
        for d in docs
    )


def build_chain(collection_name: Optional[str] = None) -> Runnable:
    settings = Settings()
    retriever = build_retriever(collection_name=collection_name)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

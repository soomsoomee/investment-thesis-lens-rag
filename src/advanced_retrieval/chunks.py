"""4주차 우승 청킹(B-large 1500/300)으로 5건 transcript를 청크화.

ingest와 BM25 retriever가 이 함수를 공유해 '같은 청크 위에서 비교'를 보장한다.
naive_rag의 loader/metadata는 그대로 재사용하고, splitter만 1500/300으로 새로 만든다
(naive_rag.splitter는 Settings의 600/100에 묶여 있어 재사용 불가).
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from advanced_retrieval.config import AdvancedSettings
from naive_rag.constants import WEEK4_CONTENT_IDS
from naive_rag.loader import load_transcripts
from naive_rag.metadata import enrich_chunks


def load_chunks() -> list[Document]:
    s = AdvancedSettings()
    docs = load_transcripts(content_ids=WEEK4_CONTENT_IDS)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=list(s.separators),
    )
    chunks = splitter.split_documents(docs)
    return enrich_chunks(chunks)

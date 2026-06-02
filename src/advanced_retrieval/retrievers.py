"""4개 retriever 빌더: Dense / BM25 / Hybrid / Hybrid+Rerank.

모두 study_advanced_retrieval 청크 위에서 동작하고 최종 k=4로 맞춘다.
"""
from kiwipiepy import Kiwi
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from advanced_retrieval.chunks import load_chunks
from advanced_retrieval.config import AdvancedSettings
from advanced_retrieval.ingest import build_vectorstore

_kiwi = Kiwi()


def kiwi_tokenize(text: str) -> list[str]:
    """BM25용 한국어 형태소 토크나이저. 공백 split보다 고유명사/조사 분리가 정확."""
    return [tok.form for tok in _kiwi.tokenize(text)]


class TopKRetriever(BaseRetriever):
    """임의 retriever 결과를 상위 k로 자르는 래퍼 (Ensemble 결과 cap용)."""

    base: BaseRetriever
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.base.invoke(query)[: self.k]


def build_dense(k: int | None = None) -> BaseRetriever:
    s = AdvancedSettings()
    vs = build_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k or s.retriever_k})


def build_bm25(k: int | None = None) -> BaseRetriever:
    s = AdvancedSettings()
    r = BM25Retriever.from_documents(load_chunks(), preprocess_func=kiwi_tokenize)
    r.k = k or s.retriever_k
    return r


def build_hybrid(final_k: int | None = None, candidate_k: int | None = None) -> BaseRetriever:
    s = AdvancedSettings()
    ck = candidate_k or s.hybrid_candidate_k
    ensemble = EnsembleRetriever(
        retrievers=[build_dense(k=ck), build_bm25(k=ck)],
        weights=list(s.ensemble_weights),
    )
    return TopKRetriever(base=ensemble, k=final_k or s.retriever_k)


def build_hybrid_rerank() -> BaseRetriever:
    s = AdvancedSettings()
    base = build_hybrid(final_k=s.rerank_candidate_k, candidate_k=s.rerank_candidate_k)
    model = HuggingFaceCrossEncoder(model_name=s.reranker_model)
    compressor = CrossEncoderReranker(model=model, top_n=s.rerank_top_n)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)

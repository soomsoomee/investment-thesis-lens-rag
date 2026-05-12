from langchain_huggingface import HuggingFaceEmbeddings

from naive_rag.config import Settings


def build_embeddings() -> HuggingFaceEmbeddings:
    settings = Settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

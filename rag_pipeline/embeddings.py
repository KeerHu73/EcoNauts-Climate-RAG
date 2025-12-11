from __future__ import annotations
from typing import Optional
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

_EMBEDDINGS: Optional[HuggingFaceBgeEmbeddings] = None

def get_embedding_model(
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: Optional[str] = None,
) -> HuggingFaceBgeEmbeddings:
    """
    Low-memory embedding loader with caching.
    """
    global _EMBEDDINGS

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDINGS

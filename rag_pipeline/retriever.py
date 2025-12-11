from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

@dataclass
class SimpleRetriever:
    vector_store: Any

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        if k < 3:
            k = 3
        return self.vector_store.similarity_search(query, k=k)

    @staticmethod
    def build_sources(docs: List[Document], max_snippet_chars: int = 220) -> List[Dict[str, Any]]:
        sources = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            content = getattr(d, "page_content", "") or ""
            sources.append({
                "id": meta.get("source", "unknown"),
                "page": meta.get("page"),
                "country": meta.get("country"),
                "year": meta.get("year"),
                "snippet": content[:max_snippet_chars],
            })
        return sources

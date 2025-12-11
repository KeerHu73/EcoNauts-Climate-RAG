from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
from .retriever import SimpleRetriever

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document


SYSTEM_PROMPT = """
You MUST answer ONLY using the provided context.
If the answer cannot be found in the context, say:
"I don't have enough information in the provided data to answer that."
""".strip()


def build_context_block(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        content = getattr(d, "page_content", "") or ""

        src = meta.get("source", "unknown")
        header_bits = [f"Source={src}"]

        page = meta.get("page")
        if page is not None:
            header_bits.append(f"page={page}")

        country = meta.get("country")
        if country:
            header_bits.append(f"country={country}")

        year = meta.get("year")
        if year:
            header_bits.append(f"year={year}")

        parts.append(f"[{i}] " + ", ".join(header_bits) + "\n" + content)

    return "\n\n".join(parts)


def dummy_llm(prompt: str, **kwargs):
    return [{
        "generated_text": prompt + "\nI don't have enough information in the provided data to answer that."
    }]


@dataclass
class RAGPipeline:
    retriever: SimpleRetriever
    llm: Any = dummy_llm

    def generate_answer(self, question: str, docs: List[Document]) -> str:
        context = build_context_block(docs)
        prompt = f"""
{SYSTEM_PROMPT}
Context:
{context}
Question:
{question}
Answer:
""".strip()

        out = self.llm(prompt, max_new_tokens=256, do_sample=False)

        if isinstance(out, list) and out and isinstance(out[0], dict):
            text = out[0].get("generated_text", "")
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text.strip()

        return str(out).strip()

    def ask(self, question: str, k: int = 3) -> Dict[str, Any]:
        docs = self.retriever.retrieve(question, k=k)
        answer = self.generate_answer(question, docs)
        sources = self.retriever.build_sources(docs)
        return {"answer": answer, "sources": sources}

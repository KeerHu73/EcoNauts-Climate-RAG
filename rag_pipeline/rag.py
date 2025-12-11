# rag_pipeline/rag.py

from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .vector_store import load_vectorstore   # Provided by your teammate



SYSTEM_PROMPT = """
You are an expert in climate science.

You must follow these rules:
- Only answer using the information contained in the provided Context.
- Do not make up facts.
- If the Context does not contain enough information, say:
  "There is not enough information in the provided data to answer that."
- When using CSV data, cite exact COUNTRY, YEAR, and VALUE.
- Keep answers concise, factual, and grounded in the data.
"""


# GLOBAL CACHES
_VECTORSTORE = None
_LLM_PIPE = None

# LOAD VECTOR STORE
def get_vectorstore():
    """Load FAISS vector store only once."""
    global _VECTORSTORE
    if _VECTORSTORE is None:
        _VECTORSTORE = load_vectorstore()
    return _VECTORSTORE


# LOAD LLM PIPELINE
def get_llm_pipeline():
    """Load the local LLM once."""
    global _LLM_PIPE

    if _LLM_PIPE is None:
        model_name = "microsoft/Phi-3-mini-4k-instruct"  # Change if needed

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        _LLM_PIPE = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=500,
            do_sample=False,
            temperature=0.1,
        )

    return _LLM_PIPE


# RETRIEVAL
def retrieve_documents(question: str, k: int = 5):
    """Retrieve top-k documents from FAISS."""
    vs = get_vectorstore()
    docs = vs.similarity_search(question, k=k)
    return docs


# BUILD CONTEXT
def build_context(docs) -> str:
    """Construct the context block sent to the LLM."""
    chunks = []
    for i, doc in enumerate(docs):
        meta = doc.metadata or {}

        country = meta.get("country", "")
        year = meta.get("year", "")
        snippet = meta.get("snippet", doc.page_content)

        header = f"[Document {i+1} | country={country}, year={year}]\n"
        chunks.append(header + snippet + "\n")

    return "\n".join(chunks)


# FORMAT SOURCE METADATA FOR API RESPONSE
def docs_to_sources(docs) -> List[Dict]:
    """Convert retrieved docs to JSON-serializable source objects."""
    sources = []

    for doc in docs:
        meta = doc.metadata or {}

        entry = {
            "id": meta.get("id", "etl_cleaned_dataset.csv"),
            "snippet": meta.get("snippet", doc.page_content[:200])
        }

        if "country" in meta:
            entry["country"] = meta["country"]
        if "year" in meta:
            entry["year"] = meta["year"]

        sources.append(entry)

    return sources


# BUILD FULL PROMPT
def build_prompt(question: str, context: str) -> str:
    """Construct the final LLM prompt including system instructions."""
    return f"""[SYSTEM]
{SYSTEM_PROMPT}

[CONTEXT]
{context}

[QUESTION]
{question}

Answer ONLY using the Context. Do not hallucinate.
"""

# GENERATE ANSWER
def generate_answer(prompt: str) -> str:
    llm = get_llm_pipeline()
    output = llm(prompt)[0]["generated_text"]

    if output.startswith(prompt):
        output = output[len(prompt):]

    return output.strip()

# MAIN ENTRY POINT (USED BY THE API)
def answer_question(question: str) -> Tuple[str, List[Dict]]:
    """Main function used by app.py to produce an answer + sources."""

    if not question.strip():
        return "Please provide a valid question.", []

    docs = retrieve_documents(question)
    context = build_context(docs)
    prompt = build_prompt(question, context)
    answer = generate_answer(prompt)
    sources = docs_to_sources(docs)

    return answer, sources

from pathlib import Path
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import get_embedding_model
from vector_store import FaissVectorStore

DATA_PATH = Path("data/etl_cleaned_dataset.csv")


def main():
    # 1. Load CSV
    df = pd.read_csv(DATA_PATH)

    documents = []
    for _, row in df.iterrows():
        text = " ".join(str(v) for v in row.values if pd.notna(v))
        documents.append(
            Document(
                page_content=text,
                metadata={"source": "etl_cleaned_dataset.csv"}
            )
        )

    print(f"Loaded {len(documents)} documents")

    # 2. Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # 3. Build embeddings + FAISS
    embedding = get_embedding_model()
    store = FaissVectorStore(embedding=embedding)
    store.add_documents(chunks)

    # 4. Save index
    store.save_local()
    print("FAISS index saved to vector_store/faiss_index")


if __name__ == "__main__":
    main()

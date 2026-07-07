"""
Document loading, chunking, and embedding for the RAG pipeline.
"""

import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Load the embedding model for converting text to vectors.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def detect_department(filename: str) -> str:
    """
    Detect department from filename.

    Convention: files are named like "finance_budget_q1.txt"
    The first part before underscore is the department.
    """
    name = os.path.basename(filename).lower()
    for dept in ["finance", "hr", "marketing", "engineering", "executive", "general"]:
        if name.startswith(dept):
            return dept
    return "general"


def load_documents(docs_dir: str) -> list:
    """
    Load all text documents from a directory.
    Returns list of dicts with text, filename, and department.
    """
    documents = []
    for filename in os.listdir(docs_dir):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r") as f:
            text = f.read().strip()

        if text:
            department = detect_department(filename)
            documents.append({
                "text": text,
                "filename": filename,
                "department": department,
                "source": filepath,
            })

    return documents


def chunk_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """
    Split documents into smaller chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": doc["filename"],
                    "department": doc["department"],
                    "chunk_index": i,
                }
            })

    return chunks


def build_vectorstore(chunks: list, persist_dir: str) -> Chroma:
    """
    Create a Chroma vector database from document chunks.
    """
    embeddings = get_embedding_model()

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Use PersistentClient for ChromaDB 1.x compatibility
    client = chromadb.PersistentClient(path=persist_dir)

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        client=client,
        collection_name="finsolve_docs",
    )

    return vectorstore


def load_vectorstore(persist_dir: str) -> Chroma:
    """Load existing vector database from disk."""
    embeddings = get_embedding_model()
    client = chromadb.PersistentClient(path=persist_dir)
    return Chroma(
        client=client,
        collection_name="finsolve_docs",
        embedding_function=embeddings,
    )

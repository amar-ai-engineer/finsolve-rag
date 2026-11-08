import os

import pytest

from src.document_loader import build_vectorstore, chunk_documents, load_documents
from tests.fakes import HashingEmbeddings

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "documents")


@pytest.fixture(scope="session")
def chunks():
    """The real company documents, chunked exactly like the app does."""
    docs = load_documents(DOCS_DIR)
    return chunk_documents(docs, chunk_size=500, chunk_overlap=100)


@pytest.fixture(scope="session")
def vectorstore(chunks, tmp_path_factory):
    """A real Chroma store built from the real chunks, with fake embeddings.

    Built once per test session in a temporary directory, so tests never
    touch data/vectorstore/ and never need the network.
    """
    persist_dir = str(tmp_path_factory.mktemp("chroma"))
    return build_vectorstore(chunks, persist_dir, embeddings=HashingEmbeddings())

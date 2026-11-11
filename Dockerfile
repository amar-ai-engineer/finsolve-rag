# One image for both services (FastAPI and Streamlit); docker-compose.yml
# picks the command. The embedding model and the vector index are baked in at
# build time, so a container starts without internet access and without a
# shared volume that two processes would write to at once.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Where sentence-transformers caches the downloaded model.
    HF_HOME=/opt/hf-cache \
    # Chroma's anonymous usage telemetry.
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# 1. Dependencies (installed as root, cached until requirements.txt changes).
# CPU-only torch first: the default PyPI wheel bundles CUDA and is several GB.
# Same version CI resolves from the CPU index.
ARG TORCH_VERSION=2.15.0
RUN pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2. Non-root user. It owns only what it must write: the model cache and data/.
# Application code below stays root-owned, so the running process cannot modify it.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /opt/hf-cache /app/data \
    && chown appuser:appuser /opt/hf-cache /app/data
USER appuser

# 3. Download the embedding model and build the Chroma index from the documents,
# using the same function the app calls at startup. Changing a document means
# rebuilding the image, which keeps the index and the documents in sync.
COPY src/ src/
COPY data/documents/ data/documents/
RUN python -c "from src.document_loader import load_or_build_vectorstore as build; \
assert build('data/documents', 'data/vectorstore') is not None, 'no documents indexed'"

# From here on, never reach out to the Hugging Face Hub: everything is local.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# 4. The rest of the application code (changes here do not rebuild the index).
COPY api/ api/
COPY app.py .
COPY .streamlit/ .streamlit/

EXPOSE 8000 8501

# Default service: the API. docker-compose.yml overrides this for the UI.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

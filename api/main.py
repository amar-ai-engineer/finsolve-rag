"""
FinSolve HTTP API.

Same RAG pipeline and RBAC rules as the Streamlit app, exposed over HTTP:

    POST /auth/login   username + password -> signed JWT (1 hour)
    POST /ask          question + Bearer token -> answer, sources, citations, chunks_retrieved
    GET  /health       liveness check

The caller's role is read ONLY from the verified token. Anything a client
puts in the request body (for example "role": "executive") is ignored.

Run locally:
    export FINSOLVE_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
    export GOOGLE_API_KEY=...
    uvicorn api.main:app --reload
"""

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.auth import (
    TOKEN_TTL_SECONDS,
    TokenError,
    authenticate,
    create_access_token,
    decode_access_token,
    get_jwt_secret,
)
from src.document_loader import load_or_build_vectorstore
from src.observability import configure_logging
from src.rag_chain import ask_question, get_llm

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(ROOT_DIR, "data", "documents")
VECTORSTORE_DIR = os.path.join(ROOT_DIR, "data", "vectorstore")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail at startup, not on the first login, if the signing secret is missing.
    get_jwt_secret()
    # One JSON log line per /ask request (role, latency, sources, tokens).
    configure_logging()
    yield


app = FastAPI(title="FinSolve API", version="1.0.0", lifespan=lifespan)

# auto_error=False so a missing header gives 401 (not FastAPI's default 403),
# with the same error shape as a bad or expired token.
bearer_scheme = HTTPBearer(auto_error=False)


# ---------- Request / response models ----------

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str


class AskRequest(BaseModel):
    # Only the question. There is deliberately no "role" field; unknown
    # fields are ignored by Pydantic, so a client cannot pick its own role.
    question: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    source: str
    department: str
    snippet: str  # first ~200 characters of the retrieved chunk


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    # One entry per retrieved chunk, in rank order.
    citations: list[Citation]
    chunks_retrieved: int


# ---------- Dependencies (overridden in tests) ----------

@lru_cache(maxsize=1)
def get_vectorstore():
    """Load (or build on first run) the Chroma store, once per process."""
    store = load_or_build_vectorstore(DOCS_DIR, VECTORSTORE_DIR)
    if store is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No documents indexed")
    return store


def get_chat_model():
    """The LLM used for answers. 503 if GOOGLE_API_KEY is not configured."""
    llm = get_llm()
    if llm is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "LLM not configured: set GOOGLE_API_KEY")
    return llm


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> dict:
    """Verify the Bearer token and return its claims (sub, role, name, ...)."""
    if credentials is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_access_token(credentials.credentials)
    except TokenError as e:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# ---------- Routes ----------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/auth/login")
def login(body: LoginRequest) -> TokenResponse:
    user = authenticate(body.username, body.password)
    if user is None:
        # Same message for unknown user and wrong password.
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenResponse(
        access_token=create_access_token(user),
        expires_in=TOKEN_TTL_SECONDS,
        role=user["role"],
    )


# Plain `def` (not async): retrieval and the LLM call are blocking, so
# FastAPI runs this in its thread pool instead of blocking the event loop.
@app.post("/ask")
def ask(
    body: AskRequest,
    user: Annotated[dict, Depends(get_current_user)],
    vectorstore: Annotated[object, Depends(get_vectorstore)],
    llm: Annotated[object, Depends(get_chat_model)],
) -> AskResponse:
    result = ask_question(
        query=body.question,
        role=user["role"],  # from the signed token, never from the body
        user_name=user.get("name", user["sub"]),
        vectorstore=vectorstore,
        llm=llm,
    )
    return AskResponse(**result)

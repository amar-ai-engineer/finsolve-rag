# FinSolve: RAG Assistant with Role-Based Access Control

> A company-knowledge chatbot where each user can only retrieve the documents their role is allowed to see. Access control is enforced **in the vector search**, before anything reaches the LLM.

## The problem

Internal RAG chatbots usually index every document in one vector store and rely on the system prompt to keep sensitive content away from the wrong people. That is fragile: the model has already seen the board minutes when it is asked to "please not mention them", and a prompt-injection attack only has to talk it out of that rule.

## The approach

FinSolve treats access control as a **retrieval problem, not a prompt problem**:

1. Every chunk is stored with a `department` metadata field (finance, hr, marketing, engineering, executive, general).
2. Each role maps to a list of allowed departments.
3. The similarity search runs with a metadata filter (`department IN allowed_departments`), so unauthorized chunks are never returned to the application, and therefore never placed in the LLM's context.
4. The system prompt adds a second, weaker layer: answer only from the provided context, cite sources, refuse role-change requests.

If the prompt layer fails, the model still has nothing sensitive to leak, because it was never given it.

```mermaid
flowchart LR
    U[User + role] --> Q[Question]
    Q --> R{{Vector search<br/>filter: department IN role.allowed}}
    R -->|only authorized chunks| C[Context + citations]
    C --> L[Gemini 2.5 Flash<br/>grounded answer]
    L --> A[Answer + sources]
    D[(ChromaDB<br/>chunks + department metadata)] --> R
```

## Roles

| Demo user | Role | Can retrieve |
|---|---|---|
| `sarah_cfo` | Finance | finance + general |
| `mike_hr` | HR | hr + general |
| `lisa_mkt` | Marketing | marketing + general |
| `raj_eng` | Engineering | engineering + general |
| `ceo` | Executive | all departments, including board minutes |
| `intern` | Employee | general only |

All demo accounts use the password `demo123`. These are **demo credentials stored in code**; see *Limitations*.

## Evaluation

Work in progress, tracked in this repo's pull requests:

- **Access-control tests** ([`tests/test_rbac.py`](tests/test_rbac.py), run in CI on every PR): for each of the 6 roles, normal, cross-department and prompt-injection queries are run against a real ChromaDB collection built from `data/documents`, and the test fails if any returned chunk belongs to a department outside the role's allowed list. Unknown roles must fall back to `general`, and the Python fallback path (store rejects metadata filters) is tested too. The tests use a deterministic hashing embedding instead of MiniLM, because the guarantee comes from the metadata filter, not from embedding quality, and this keeps them offline.
- **Retrieval quality**: a golden set of questions with known source documents, reporting hit@k and MRR (how often, and how high, the right document is retrieved).

Numbers will be added here only once they are produced by the evaluation scripts in this repository.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangChain (core, text splitters) | Simple, well-known RAG building blocks |
| Vector store | ChromaDB (local, persistent) | Metadata filtering built in, no server to run |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Small, fast on CPU, free |
| LLM | Google Gemini 2.5 Flash | Low cost, good instruction following |
| UI | Streamlit | Login + chat interface in one file |

## Run it

```bash
git clone https://github.com/amar-ai-engineer/finsolve-rag.git
cd finsolve-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GOOGLE_API_KEY
streamlit run app.py
```

The first run downloads the embedding model and builds the vector store in `data/vectorstore/`.

### Run the tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The tests do not need an API key or the embedding model download.

## Project structure

```
finsolve-rag/
├── app.py                  # Streamlit UI: login + chat
├── src/
│   ├── auth.py             # Roles, allowed departments, demo login
│   ├── document_loader.py  # Load, chunk, embed, persist to ChromaDB
│   └── rag_chain.py        # RBAC-filtered retrieval + grounded generation
├── data/documents/         # 12 sample company documents across 6 departments
├── tests/                  # RBAC leak tests (offline, fake embeddings)
├── .github/workflows/ci.yml  # ruff + pytest on every push and PR
├── requirements.txt        # Pinned runtime dependencies
└── requirements-dev.txt    # + test dependencies
```

## Design decisions

- **Filter in the database, not after retrieval.** Post-filtering (retrieve top-k, then drop unauthorized chunks) can silently return fewer than k results, or none, for restricted roles. Filtering inside the search keeps k authorized results.
- **Department from file naming convention.** Simple and transparent for a demo. A production system would take it from the document management system's ACLs.
- **Short chat memory (last 3 exchanges).** Keeps follow-up questions working without letting the context grow without limit.
- **Low temperature (0.3).** Favors answers that stay close to the retrieved text.

## Limitations

- Demo authentication only: plaintext passwords in code, no sessions or tokens. A production version needs a real identity provider (OIDC/SSO) or at least hashed passwords and signed tokens.
- Access rules are per department. Real organizations also need per-document and per-user permissions.
- Sample documents are synthetic and small (12 files), so retrieval quality numbers will not transfer directly to a real corpus.
- No rate limiting, audit logging or PII redaction yet.

## Built by

**Amar Ismail**, AI Engineer · [Portfolio](https://amar-ai-engineer.github.io) · [LinkedIn](https://www.linkedin.com/in/amar-ai-engineer/) · [GitHub](https://github.com/amar-ai-engineer)

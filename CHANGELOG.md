# Changelog

All notable changes to this project. Each step was one reviewed pull request.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - unreleased (tag after Day 7 is merged)

### Added
- **Structured request logging** (Day 7): one JSON line per question with role, retrieval mode, outcome, latency (total, retrieval, LLM), chunks retrieved, sources, model and token usage. The question text is excluded by default (`FINSOLVE_LOG_QUESTIONS=1` to opt in).
- **Citations with snippets** (Day 7): `ask_question()` and `POST /ask` return a `citations` list (source, department, ~200-character snippet per retrieved chunk); the Streamlit UI shows them in an expander under each answer.
- **Hybrid retrieval** (Day 6, [#6](https://github.com/amar-ai-engineer/finsolve-rag/pull/6)): BM25 keyword search plus vector search, merged with reciprocal rank fusion, with the role filter applied inside each search before fusion. Selected with `FINSOLVE_RETRIEVAL_MODE`; `hybrid` became the default because it scored higher on the retrieval eval. Leak tests run in both modes.
- **Docker** (Day 5, [#5](https://github.com/amar-ai-engineer/finsolve-rag/pull/5)): one image (CPU torch, non-root user, embedding model and index baked in) and a docker-compose file with `api` and `ui` services and healthchecks. CI builds the image and smoke-tests it.
- **FastAPI service** (Day 4, [#4](https://github.com/amar-ai-engineer/finsolve-rag/pull/4)): `POST /auth/login` (signed JWT, 1 hour), `POST /ask` (role read from the token only), `GET /health`.
- **Retrieval evaluation** (Day 3, [#3](https://github.com/amar-ai-engineer/finsolve-rag/pull/3)): a 30-question golden set and `eval/run_eval.py` (hit@1, hit@5, MRR, leak rate per role) run with the real embedding model in a GitHub Actions workflow.
- **Access-control test suite and CI** (Day 2, [#2](https://github.com/amar-ai-engineer/finsolve-rag/pull/2)): every role is checked against normal, cross-department and prompt-injection queries; ruff and pytest run on every push and PR.

### Changed
- **Passwords** (Day 4): demo passwords are stored as bcrypt hashes instead of plaintext; the Streamlit app and the API share one auth module.
- **Testability** (Day 2): the embedding model and Chroma client can be injected, so tests run offline with a deterministic fake embedding.
- **README and dependencies** (Day 1, [#1](https://github.com/amar-ai-engineer/finsolve-rag/pull/1)): rewritten README that describes only what the code does, pinned dependency versions, and migration from deprecated LangChain imports to `langchain-chroma` and `langchain-huggingface`.

## Initial version - 2026-07-08 (untagged)

### Added
- First version: Streamlit chat app with demo logins for six roles, ChromaDB index with a `department` field per chunk, role-filtered similarity search, and answers from Gemini 2.5 Flash.

"""
Retrieval evaluation for FinSolve.

Runs every question in eval/golden_set.jsonl through the same RBAC-filtered
retrieval the app uses (src.rag_chain.retrieve_with_rbac) and reports:

- hit@1 / hit@5: how often the expected source document is the first
  retrieved chunk / anywhere in the top 5 chunks.
- MRR (mean reciprocal rank): 1 / rank of the first chunk from the expected
  source, averaged over questions (0 when it is not retrieved at all).
- leak rate: share of ALL questions where at least one retrieved chunk
  belongs to a department the role may not see. This must be 0.

Questions with "expected_source": null are ones the role must NOT be able to
answer (the answer lives only in documents outside its departments). They are
excluded from hit@k and MRR and only count towards the leak rate.

Each retrieval mode (vector, hybrid) is evaluated on the same index, so the
results table is a like-for-like comparison.

Only retrieval is evaluated, not the LLM's answer, so no API key is needed.

Usage (from the repo root):
    python -m eval.run_eval                       # real all-MiniLM-L6-v2 model, both modes
    python -m eval.run_eval --modes hybrid        # one mode only
    python -m eval.run_eval --embeddings hashing  # offline smoke test, numbers are meaningless
"""

import argparse
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from datetime import UTC, datetime

from src.auth import ROLES, get_allowed_departments
from src.document_loader import build_vectorstore, chunk_documents, load_documents
from src.rag_chain import RETRIEVAL_MODES, retrieve_with_rbac

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(REPO_ROOT, "data", "documents")
GOLDEN_PATH = os.path.join(REPO_ROOT, "eval", "golden_set.jsonl")
OUT_DIR = os.path.join(REPO_ROOT, "eval")

# Same values the app uses (app.py / ask_question), so we measure what users get.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5


def load_golden(path: str = GOLDEN_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def first_rank(sources: list[str], expected: str) -> int | None:
    """1-based position of the first chunk from `expected`, or None if absent."""
    for i, source in enumerate(sources, 1):
        if source == expected:
            return i
    return None


def evaluate(golden: list[dict], vectorstore, k: int = TOP_K, mode: str = "vector") -> list[dict]:
    """Run each golden question through RBAC retrieval and record what came back."""
    rows = []
    for item in golden:
        allowed = get_allowed_departments(item["role"])
        docs = retrieve_with_rbac(item["question"], item["role"], vectorstore, k=k, mode=mode)
        sources = [d.metadata.get("source") for d in docs]
        leaked = [d.metadata.get("source") for d in docs if d.metadata.get("department") not in allowed]
        expected = item.get("expected_source")
        rows.append({
            "id": item["id"],
            "role": item["role"],
            "question": item["question"],
            "expected_source": expected,
            "retrieved_sources": sources,
            "rank": first_rank(sources, expected) if expected else None,
            "leaked_sources": leaked,
        })
    return rows


def _scores(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["expected_source"]]
    n = len(answerable)
    ranks = [r["rank"] for r in answerable]
    return {
        "questions": len(rows),
        "answerable": n,
        "hit@1": sum(1 for r in ranks if r == 1) / n if n else None,
        f"hit@{TOP_K}": sum(1 for r in ranks if r is not None) / n if n else None,
        "mrr": sum(1 / r for r in ranks if r is not None) / n if n else None,
        "leak_rate": sum(1 for r in rows if r["leaked_sources"]) / len(rows) if rows else None,
    }


def summarize(rows: list[dict]) -> dict:
    by_role = defaultdict(list)
    for r in rows:
        by_role[r["role"]].append(r)
    return {
        "overall": _scores(rows),
        "per_role": {role: _scores(by_role[role]) for role in ROLES if role in by_role},
    }


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _score_row(name: str, s: dict) -> str:
    k = f"hit@{TOP_K}"
    return (
        f"| {name} | {s['questions']} | {s['answerable']} | {_fmt(s['hit@1'])} | "
        f"{_fmt(s[k])} | {_fmt(s['mrr'])} | {_fmt(s['leak_rate'])} |"
    )


def to_markdown(results: dict[str, dict], meta: dict) -> str:
    """results maps mode -> {"summary": ..., "questions": rows}."""
    k = f"hit@{TOP_K}"
    header = [
        f"| Scope | Questions | Answerable | hit@1 | {k} | MRR | Leak rate |",
        "|---|---|---|---|---|---|---|",
    ]
    first = next(iter(results.values()))
    n_rows, n_answerable = len(first["questions"]), first["summary"]["overall"]["answerable"]
    lines = [
        "# Retrieval evaluation results",
        "",
        f"Generated by `python -m eval.run_eval` on {meta['generated_at']}"
        + (f" (commit `{meta['commit'][:7]}`)" if meta.get("commit") else "") + ".",
        "",
        f"- Embeddings: `{meta['embeddings']}`",
        f"- Chunking: {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap · top-k: {TOP_K}",
        f"- Retrieval modes: {', '.join(f'`{m}`' for m in results)} (same index, same golden set)",
        f"- Golden set: {n_rows} questions ({n_answerable} answerable, "
        f"{n_rows - n_answerable} that the role must not be able to answer)",
        "",
        "## Overall, by retrieval mode",
        "",
        *header,
    ]
    lines += [_score_row(f"**{mode}**", r["summary"]["overall"]) for mode, r in results.items()]

    for mode, r in results.items():
        lines += ["", f"## Per role: `{mode}`", "", *header]
        lines += [_score_row(role, s) for role, s in r["summary"]["per_role"].items()]

    modes = list(results)
    lines += [
        "",
        "## Per question",
        "",
        "Rank of the first chunk from the expected source in each mode (`miss` = not in the top-k).",
        "",
        "| ID | Role | Expected source | " + " | ".join(f"Rank ({m})" for m in modes) + " |",
        "|---|---|---|" + "---|" * len(modes),
    ]
    for i, row in enumerate(first["questions"]):
        expected = row["expected_source"] or "*(must not answer)*"
        ranks = []
        for mode in modes:
            r = results[mode]["questions"][i]
            ranks.append("-" if not r["expected_source"] else (str(r["rank"]) if r["rank"] else "miss"))
        lines.append(f"| {row['id']} | {row['role']} | {expected} | " + " | ".join(ranks) + " |")

    for mode in modes:
        lines += ["", f"### Retrieved sources: `{mode}`", "", "| ID | Retrieved (top-k, in order) |", "|---|---|"]
        lines += [f"| {r['id']} | {', '.join(r['retrieved_sources'])} |" for r in results[mode]["questions"]]
    return "\n".join(lines) + "\n"


def _git_commit() -> str | None:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", choices=["minilm", "hashing"], default="minilm",
                        help="minilm = the real model the app uses; hashing = offline fake for smoke tests")
    parser.add_argument("--modes", default=",".join(RETRIEVAL_MODES),
                        help=f"comma-separated retrieval modes to evaluate (default: {','.join(RETRIEVAL_MODES)})")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(modes) - set(RETRIEVAL_MODES)
    if not modes or unknown:
        parser.error(f"--modes must be a subset of {RETRIEVAL_MODES}, got {args.modes!r}")

    if args.embeddings == "hashing":
        from tests.fakes import HashingEmbeddings
        embeddings, name = HashingEmbeddings(), "hashing (offline fake, numbers are NOT meaningful)"
    else:
        from src.document_loader import get_embedding_model
        embeddings, name = get_embedding_model(), "all-MiniLM-L6-v2"

    chunks = chunk_documents(load_documents(DOCS_DIR), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    golden = load_golden()
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        # Fresh store in a temp dir so the eval never reads or writes data/vectorstore/.
        vectorstore = build_vectorstore(chunks, tmp, embeddings=embeddings)
        for mode in modes:
            rows = evaluate(golden, vectorstore, mode=mode)
            results[mode] = {"summary": summarize(rows), "questions": rows}

    meta = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": _git_commit(),
        "embeddings": name,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "modes": modes,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
        f.write("\n")
    markdown = to_markdown(results, meta)
    with open(os.path.join(args.out_dir, "results.md"), "w") as f:
        f.write(markdown)
    print(markdown)

    # Fail loudly (non-zero exit, red CI) if any mode retrieved an unauthorized chunk.
    leaking = [mode for mode, r in results.items() if r["summary"]["overall"]["leak_rate"]]
    if leaking:
        print(f"ERROR: leak rate is not 0 in mode(s) {leaking} - a role retrieved a chunk it is not allowed to see.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

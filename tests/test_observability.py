"""
Tests for the structured request log and the citations returned by ask_question.

Offline: real Chroma store with fake embeddings (conftest.py) and LangChain's
fake chat model.
"""

import json
import logging

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from src.observability import REQUEST_LOGGER_NAME, JsonFormatter, configure_logging
from src.rag_chain import ask_question, make_snippet

QUESTION = "What is the PTO policy for my secret medical leave?"


class UsageChatModel(FakeListChatModel):
    """Fake chat model that also reports token usage, like Gemini does."""

    def invoke(self, input, config=None, **kwargs):
        text = super().invoke(input, config, **kwargs).content
        return AIMessage(
            content=text,
            usage_metadata={"input_tokens": 120, "output_tokens": 8, "total_tokens": 128},
        )


class BrokenChatModel(FakeListChatModel):
    def invoke(self, input, config=None, **kwargs):
        raise RuntimeError(f"upstream failure while answering: {QUESTION}")


def request_logs(caplog) -> list[dict]:
    """The structured fields of every rag_request log record."""
    return [r.fields for r in caplog.records if r.name == REQUEST_LOGGER_NAME]


@pytest.fixture
def capture(caplog, monkeypatch):
    monkeypatch.delenv("FINSOLVE_LOG_QUESTIONS", raising=False)
    caplog.set_level(logging.INFO, logger=REQUEST_LOGGER_NAME)
    # configure_logging() turns propagation off; caplog listens on the root
    # logger, so make sure records still reach it in these tests.
    monkeypatch.setattr(logging.getLogger("finsolve"), "propagate", True)
    return caplog


def ask(vectorstore, llm, role="hr"):
    return ask_question(QUESTION, role=role, user_name="Test", vectorstore=vectorstore, llm=llm)


# ---------- the log line ----------

def test_one_log_line_per_request_with_expected_fields(vectorstore, capture):
    result = ask(vectorstore, UsageChatModel(responses=["ok"]))

    logs = request_logs(capture)
    assert len(logs) == 1
    entry = logs[0]
    assert entry["event"] == "rag_request"
    assert entry["role"] == "hr"
    assert entry["outcome"] == "answered"
    assert entry["retrieval_mode"] in ("vector", "hybrid")
    assert entry["model"] == "UsageChatModel"
    assert entry["chunks_retrieved"] == result["chunks_retrieved"] > 0
    assert entry["sources"] == result["sources"]
    for key in ("latency_ms", "retrieval_ms", "llm_ms"):
        assert isinstance(entry[key], int) and entry[key] >= 0
    assert entry["latency_ms"] >= entry["retrieval_ms"]


def test_token_usage_is_logged_when_the_model_reports_it(vectorstore, capture):
    ask(vectorstore, UsageChatModel(responses=["ok"]))
    entry = request_logs(capture)[0]
    assert (entry["input_tokens"], entry["output_tokens"], entry["total_tokens"]) == (120, 8, 128)


def test_token_usage_is_omitted_when_the_model_does_not_report_it(vectorstore, capture):
    ask(vectorstore, FakeListChatModel(responses=["ok"]))
    assert "total_tokens" not in request_logs(capture)[0]


def test_question_text_is_not_logged_by_default(vectorstore, capture):
    ask(vectorstore, FakeListChatModel(responses=["ok"]))
    entry = request_logs(capture)[0]
    assert "question" not in entry
    assert "secret medical leave" not in json.dumps(entry)


def test_question_text_is_logged_only_when_opted_in(vectorstore, capture, monkeypatch):
    monkeypatch.setenv("FINSOLVE_LOG_QUESTIONS", "1")
    ask(vectorstore, FakeListChatModel(responses=["ok"]))
    assert request_logs(capture)[0]["question"] == QUESTION


def test_llm_error_logs_type_but_not_message(vectorstore, capture):
    result = ask(vectorstore, BrokenChatModel(responses=["unused"]))
    entry = request_logs(capture)[0]
    assert entry["outcome"] == "llm_error"
    assert entry["error_type"] == "RuntimeError"
    # The exception message contains the question; it must not reach the log.
    assert "secret medical leave" not in json.dumps(entry)
    assert result["answer"].startswith("Sorry")


def test_no_llm_is_logged(vectorstore, capture, monkeypatch):
    monkeypatch.setattr("src.rag_chain.get_llm", lambda: None)
    ask_question(QUESTION, role="hr", user_name="Test", vectorstore=vectorstore)
    entry = request_logs(capture)[0]
    assert entry["outcome"] == "no_llm"
    assert entry["chunks_retrieved"] == 0


def test_json_formatter_writes_one_json_object_per_line():
    record = logging.LogRecord(REQUEST_LOGGER_NAME, logging.INFO, __file__, 1, "", None, None)
    record.fields = {"event": "rag_request", "role": "hr", "latency_ms": 12}
    line = JsonFormatter().format(record)
    assert "\n" not in line
    parsed = json.loads(line)
    assert parsed["event"] == "rag_request"
    assert parsed["role"] == "hr"
    assert parsed["level"] == "INFO"
    assert parsed["ts"].endswith("Z")


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging()
    handlers = [h for h in logging.getLogger("finsolve").handlers if getattr(h, "_finsolve_json", False)]
    assert len(handlers) == 1


# ---------- citations ----------

def test_citations_match_retrieved_chunks_and_respect_rbac(vectorstore, capture):
    result = ask(vectorstore, FakeListChatModel(responses=["ok"]), role="employee")
    citations = result["citations"]
    assert len(citations) == result["chunks_retrieved"] > 0
    assert {c["department"] for c in citations} <= {"general"}
    assert {c["source"] for c in citations} == set(result["sources"])
    for c in citations:
        assert c["snippet"]
        assert len(c["snippet"]) <= 201  # 200 characters + "…"


def test_make_snippet_collapses_whitespace_and_cuts_at_a_word():
    assert make_snippet("short\n\n  text") == "short text"
    snippet = make_snippet("word " * 100, max_chars=22)
    assert snippet == "word word word word…"

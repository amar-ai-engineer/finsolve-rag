"""
Structured request logging for FinSolve.

Every question answered by ask_question() produces ONE log line in JSON, so
it can be searched and aggregated by a log tool (Loki, CloudWatch, Datadog,
or just `jq`) without parsing free text:

    {"ts": "2026-11-13T09:12:01.123Z", "level": "INFO", "logger": "finsolve.request",
     "event": "rag_request", "role": "finance", "retrieval_mode": "hybrid",
     "outcome": "answered", "latency_ms": 812, "retrieval_ms": 41, "llm_ms": 770,
     "chunks_retrieved": 5, "sources": ["finance_q1_report.txt"],
     "model": "gemini-2.5-flash", "input_tokens": 1234, "output_tokens": 87, "total_tokens": 1321}

Privacy: the question text is NOT logged by default, because users may type
personal or confidential details into it. Set FINSOLVE_LOG_QUESTIONS=1 to
include it (for local debugging only).
"""

import json
import logging
import os
import sys
from datetime import UTC, datetime

REQUEST_LOGGER_NAME = "finsolve.request"
LOG_QUESTIONS_ENV = "FINSOLVE_LOG_QUESTIONS"

request_logger = logging.getLogger(REQUEST_LOGGER_NAME)


class JsonFormatter(logging.Formatter):
    """Format a log record as one JSON object per line.

    Structured fields are passed with `extra={"fields": {...}}` and merged
    into the top level of the JSON object.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
        }
        message = record.getMessage()
        if message:
            payload["message"] = message
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # default=str: never let an odd value (e.g. a Path) break logging.
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> None:
    """Send FinSolve's logs to stderr as JSON lines.

    Safe to call more than once (Streamlit re-runs the script on every
    interaction): the handler is only added the first time. Only the
    "finsolve" logger tree is configured, so other libraries keep their
    own log format.
    """
    logger = logging.getLogger("finsolve")
    logger.setLevel(level or os.getenv("FINSOLVE_LOG_LEVEL", "INFO"))
    if not any(getattr(h, "_finsolve_json", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        handler._finsolve_json = True
        logger.addHandler(handler)
    # Don't also print through the root logger (would duplicate lines).
    logger.propagate = False


def should_log_questions() -> bool:
    return os.getenv(LOG_QUESTIONS_ENV, "").strip().lower() in {"1", "true", "yes"}


def log_request(fields: dict, question: str | None = None) -> None:
    """Write one structured "rag_request" log line.

    `question` is only added when FINSOLVE_LOG_QUESTIONS is enabled.
    """
    fields = {"event": "rag_request", **fields}
    if question is not None and should_log_questions():
        fields["question"] = question
    request_logger.info("", extra={"fields": fields})

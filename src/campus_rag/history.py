from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .retrieval import SearchResult


def read_jsonl(path: Path) -> list[dict]:
    """Read valid records from a local JSONL file without failing on a bad line."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


class AnswerHistory:
    """Append-only local history for inspecting real user questions and failures."""

    def __init__(self, path: Path):
        self.path = path

    def append(
        self,
        question: str,
        answer: str,
        evidence: list[SearchResult],
        latency_ms: float,
        trace: dict | None = None,
    ) -> str:
        trace_id = str(uuid4())
        record = {
            "id": trace_id,
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": answer,
            "sources": list(dict.fromkeys(item.source for item in evidence)),
            "latency_ms": round(latency_ms, 1),
            "retrieval": [
                {"rank": rank, "source": item.source, "context": item.context, "score": round(item.score, 6)}
                for rank, item in enumerate(evidence, 1)
            ],
            "trace": trace or {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return trace_id

    def latest(self, limit: int = 20) -> list[dict]:
        return list(reversed(read_jsonl(self.path)[-limit:]))

    def all(self) -> list[dict]:
        return read_jsonl(self.path)


class FeedbackHistory:
    """Append-only local user feedback linked to an answer history record."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, answer_id: str, rating: str, reason: str | None = None, note: str = "") -> dict:
        record = {
            "answer_id": answer_id,
            "rating": rating,
            "reason": reason,
            "note": note,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def latest(self, limit: int = 20) -> list[dict]:
        return list(reversed(read_jsonl(self.path)[-limit:]))

    def all(self) -> list[dict]:
        return read_jsonl(self.path)

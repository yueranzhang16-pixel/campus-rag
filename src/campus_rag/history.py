from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .retrieval import SearchResult


class AnswerHistory:
    """Append-only local history for inspecting real user questions and failures."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, question: str, answer: str, evidence: list[SearchResult], latency_ms: float) -> str:
        record = {
            "id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": answer,
            "sources": list(dict.fromkeys(item.source for item in evidence)),
            "latency_ms": round(latency_ms, 1),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record["id"]

    def latest(self, limit: int = 20) -> list[dict]:
        if not self.path.is_file():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(reversed(records[-limit:]))


class FeedbackHistory:
    """Append-only local user feedback linked to an answer history record."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, answer_id: str, rating: str) -> dict:
        record = {
            "answer_id": answer_id,
            "rating": rating,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def latest(self, limit: int = 20) -> list[dict]:
        if not self.path.is_file():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(reversed(records[-limit:]))

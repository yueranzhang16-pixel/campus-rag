from __future__ import annotations

import json
from pathlib import Path

from .corpus import build_corpus_manifest


def get_index_freshness(index_path: Path, docs_dir: Path) -> str:
    if not index_path.is_file():
        return "missing"
    if not docs_dir.is_dir():
        return "unknown"
    try:
        stored = json.loads(index_path.read_text(encoding="utf-8")).get("corpus_fingerprint")
    except (OSError, json.JSONDecodeError):
        return "unknown"
    if not stored:
        return "unknown"
    current = build_corpus_manifest(docs_dir)["fingerprint"]
    return "fresh" if stored == current else "stale"

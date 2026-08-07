from __future__ import annotations

import hashlib
import json
from pathlib import Path


TEXT_SUFFIXES = {".md", ".txt"}


def corpus_files(docs_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(docs_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    ]


def build_corpus_manifest(docs_dir: Path) -> dict:
    """Create a deterministic fingerprint of all retrievable source files."""
    files = []
    for path in corpus_files(docs_dir):
        content = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(docs_dir).as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    canonical = json.dumps(files, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "fingerprint": hashlib.sha256(canonical).hexdigest(),
        "file_count": len(files),
        "files": files,
    }

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .corpus import corpus_files
from .retrieval import load_parent_child_corpus


INSTRUCTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|above)\s+(?:instructions|rules)", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"忽略.{0,16}(?:之前|以上|前面).{0,16}(?:指令|要求)"),
)


def find_instruction_like_lines(text: str) -> list[dict]:
    findings = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if any(pattern.search(line) for pattern in INSTRUCTION_PATTERNS):
            findings.append({"line": line_number, "text": line.strip()[:240]})
    return findings


def lint_documents(docs_dir: Path, max_chunk_chars: int = 2400) -> dict:
    """Report ingestion risks without mutating a user's source documents."""
    paths = corpus_files(docs_dir)
    empty_documents = []
    instruction_like_lines = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            empty_documents.append(path.relative_to(docs_dir).as_posix())
        for finding in find_instruction_like_lines(text):
            instruction_like_lines.append({"path": path.relative_to(docs_dir).as_posix(), **finding})
    name_counts = Counter(path.name for path in paths)
    duplicate_filenames = sorted(name for name, count in name_counts.items() if count > 1)
    chunks, _ = load_parent_child_corpus(docs_dir)
    oversized_chunks = [
        {"source": chunk.source, "context": chunk.context, "chars": len(chunk.text)}
        for chunk in chunks
        if len(chunk.text) > max_chunk_chars
    ]
    return {
        "documents": len(paths),
        "chunks": len(chunks),
        "max_chunk_chars": max_chunk_chars,
        "empty_documents": empty_documents,
        "duplicate_filenames": duplicate_filenames,
        "oversized_chunks": oversized_chunks,
        "instruction_like_lines": instruction_like_lines,
        "warning_count": len(empty_documents) + len(duplicate_filenames) + len(oversized_chunks) + len(instruction_like_lines),
    }

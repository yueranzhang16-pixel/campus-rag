from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(?!include\b|define\b)(.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str
    context: str = ""
    parent_id: str = ""
    parent_position: int = 0


@dataclass(frozen=True)
class ParentChunk:
    """A section-level parent made of smaller retrievable child passages."""

    id: str
    source: str
    context: str
    segments: list[str]


@dataclass(frozen=True)
class SearchResult:
    source: str
    text: str
    score: float
    context: str = ""
    parent_text: str = ""


def tokenize(text: str) -> list[str]:
    """Tokenize English words and individual Chinese characters without dependencies."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def split_markdown_document(source: str, text: str) -> tuple[list[Chunk], dict[str, ParentChunk]]:
    """Create retrievable child passages and section-level parents for context expansion."""
    pending: list[tuple[str, str, str]] = []
    headings: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        clean = "\n".join(buffer).strip()
        buffer.clear()
        if not clean:
            return
        context = " ".join([Path(source).stem, *(title for _, title in headings)])
        parent_titles = [title for level, title in headings if level <= 2]
        parent_context = " ".join([Path(source).stem, *parent_titles]) or Path(source).stem
        parent_id = f"{source}::{parent_context}"
        pending.append((clean, context, parent_id))

    for line in text.splitlines():
        match = HEADING_PATTERN.fullmatch(line.strip())
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            headings[:] = [(old_level, old_title) for old_level, old_title in headings if old_level < level]
            headings.append((level, title))
        elif not line.strip():
            flush()
        elif set(line.strip()) <= {"-", "_", "*"}:
            flush()
        else:
            buffer.append(line)
    flush()

    parent_segments: dict[str, list[str]] = {}
    parent_metadata: dict[str, tuple[str, str]] = {}
    for clean, context, parent_id in pending:
        parent_segments.setdefault(parent_id, []).append(clean)
        parent_metadata.setdefault(parent_id, (source, " ".join(context.split()[:3]) if context else Path(source).stem))
    parents = {
        parent_id: ParentChunk(
            id=parent_id,
            source=parent_metadata[parent_id][0],
            context=parent_metadata[parent_id][1],
            segments=segments,
        )
        for parent_id, segments in parent_segments.items()
    }
    positions: dict[str, int] = {}
    chunks = []
    for clean, context, parent_id in pending:
        position = positions.get(parent_id, 0)
        chunks.append(Chunk(source, clean, context, parent_id, position))
        positions[parent_id] = position + 1
    return chunks, parents


def split_markdown_chunks(source: str, text: str) -> list[Chunk]:
    """Backward-compatible child-only view of Markdown splitting."""
    return split_markdown_document(source, text)[0]


def load_parent_child_corpus(docs_dir: Path) -> tuple[list[Chunk], dict[str, ParentChunk]]:
    chunks: list[Chunk] = []
    parents: dict[str, ParentChunk] = {}
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        file_chunks, file_parents = split_markdown_document(path.name, path.read_text(encoding="utf-8").strip())
        chunks.extend(file_chunks)
        parents.update(file_parents)
    if not chunks:
        raise ValueError(f"No .md or .txt content found under {docs_dir}")
    return chunks, parents


def load_chunks(docs_dir: Path) -> list[Chunk]:
    return load_parent_child_corpus(docs_dir)[0]


class TfidfIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        document_frequency: Counter[str],
        parents: dict[str, ParentChunk] | None = None,
        corpus_fingerprint: str | None = None,
    ):
        self.chunks = chunks
        self.document_frequency = document_frequency
        self.parents = parents or {}
        self.corpus_fingerprint = corpus_fingerprint

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        parents: dict[str, ParentChunk] | None = None,
        corpus_fingerprint: str | None = None,
    ) -> "TfidfIndex":
        df: Counter[str] = Counter()
        for chunk in chunks:
            df.update(set(tokenize(cls._searchable_text(chunk))))
        return cls(chunks, df, parents, corpus_fingerprint)

    @staticmethod
    def _searchable_text(chunk: Chunk) -> str:
        """Index passage content together with its document and section metadata."""
        return f"{chunk.context}\n{chunk.text}".strip()

    def _idf(self, token: str) -> float:
        return math.log((len(self.chunks) + 1) / (self.document_frequency[token] + 1)) + 1

    def _parent_window(self, chunk: Chunk, radius: int = 2) -> str:
        parent = self.parents.get(chunk.parent_id)
        if not parent:
            return ""
        start = max(0, chunk.parent_position - radius)
        end = min(len(parent.segments), chunk.parent_position + radius + 1)
        window = parent.segments[start:end]
        return "\n\n".join(segment for segment in window if segment != chunk.text)

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        query = Counter(tokenize(question))
        if not query:
            return []
        scored: list[SearchResult] = []
        for chunk in self.chunks:
            doc = Counter(tokenize(self._searchable_text(chunk)))
            dot = sum(query[token] * doc[token] * self._idf(token) ** 2 for token in query)
            q_norm = math.sqrt(sum((count * self._idf(token)) ** 2 for token, count in query.items()))
            d_norm = math.sqrt(sum((count * self._idf(token)) ** 2 for token, count in doc.items()))
            score = dot / (q_norm * d_norm) if q_norm and d_norm else 0.0
            scored.append(SearchResult(chunk.source, chunk.text, round(score, 4), chunk.context, self._parent_window(chunk)))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def to_dict(self) -> dict:
        return {
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "document_frequency": dict(self.document_frequency),
            "parents": [asdict(parent) for parent in self.parents.values()],
            "corpus_fingerprint": self.corpus_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TfidfIndex":
        return cls(
            chunks=[Chunk(**chunk) for chunk in data["chunks"]],
            document_frequency=Counter(data["document_frequency"]),
            parents={parent["id"]: ParentChunk(**parent) for parent in data.get("parents", [])},
            corpus_fingerprint=data.get("corpus_fingerprint"),
        )

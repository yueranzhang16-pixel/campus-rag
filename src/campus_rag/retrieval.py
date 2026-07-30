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


@dataclass(frozen=True)
class SearchResult:
    source: str
    text: str
    score: float
    context: str = ""


def tokenize(text: str) -> list[str]:
    """Tokenize English words and individual Chinese characters without dependencies."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def split_markdown_chunks(source: str, text: str) -> list[Chunk]:
    """Split content by paragraphs while carrying the complete Markdown heading path."""
    chunks: list[Chunk] = []
    headings: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        clean = "\n".join(buffer).strip()
        buffer.clear()
        if not clean:
            return
        context = " ".join([Path(source).stem, *(title for _, title in headings)])
        chunks.append(Chunk(source=source, text=clean, context=context))

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
    return chunks


def load_chunks(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        chunks.extend(split_markdown_chunks(path.name, path.read_text(encoding="utf-8").strip()))
    if not chunks:
        raise ValueError(f"No .md or .txt content found under {docs_dir}")
    return chunks


class TfidfIndex:
    def __init__(self, chunks: list[Chunk], document_frequency: Counter[str]):
        self.chunks = chunks
        self.document_frequency = document_frequency

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "TfidfIndex":
        df: Counter[str] = Counter()
        for chunk in chunks:
            df.update(set(tokenize(cls._searchable_text(chunk))))
        return cls(chunks, df)

    @staticmethod
    def _searchable_text(chunk: Chunk) -> str:
        """Index passage content together with its document and section metadata."""
        return f"{chunk.context}\n{chunk.text}".strip()

    def _idf(self, token: str) -> float:
        return math.log((len(self.chunks) + 1) / (self.document_frequency[token] + 1)) + 1

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
            scored.append(SearchResult(chunk.source, chunk.text, round(score, 4), chunk.context))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def to_dict(self) -> dict:
        return {
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "document_frequency": dict(self.document_frequency),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TfidfIndex":
        return cls(
            chunks=[Chunk(**chunk) for chunk in data["chunks"]],
            document_frequency=Counter(data["document_frequency"]),
        )

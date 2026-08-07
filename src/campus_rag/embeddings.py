from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .retrieval import Chunk, ParentChunk, SearchResult, TfidfIndex

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@dataclass
class EmbeddingIndex:
    """A CPU-friendly dense-vector index for semantic retrieval experiments."""

    chunks: list[Chunk]
    vectors: np.ndarray
    model_name: str = DEFAULT_MODEL
    parents: dict[str, ParentChunk] = field(default_factory=dict)
    corpus_fingerprint: str | None = None
    _encoder: Any | None = None

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        model_name: str = DEFAULT_MODEL,
        parents: dict[str, ParentChunk] | None = None,
        corpus_fingerprint: str | None = None,
    ) -> "EmbeddingIndex":
        index = cls(
            chunks=chunks,
            vectors=np.empty((0, 0)),
            model_name=model_name,
            parents=parents or {},
            corpus_fingerprint=corpus_fingerprint,
        )
        encoder = index._get_encoder()
        passages = [TfidfIndex._searchable_text(chunk) for chunk in chunks]
        index.vectors = np.asarray(
            encoder.encode(passages, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32
        )
        return index

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "缺少 sentence-transformers。请先运行：python -m pip install sentence-transformers"
                ) from exc
            # The model is downloaded during `embedding-index`. Subsequent queries
            # must stay offline so transient Hub connectivity cannot break serving.
            self._encoder = SentenceTransformer(self.model_name, device="cpu", local_files_only=True)
        return self._encoder

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        if not question.strip() or len(self.chunks) == 0:
            return []
        query = QUERY_INSTRUCTION + question
        vector = np.asarray(
            self._get_encoder().encode([query], normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )[0]
        scores = self.vectors @ vector
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                source=self.chunks[position].source,
                text=self.chunks[position].text,
                context=self.chunks[position].context,
                score=round(float(scores[position]), 4),
                parent_text=self._parent_window(self.chunks[position]),
            )
            for position in ranked
        ]

    def _parent_window(self, chunk: Chunk, radius: int = 2) -> str:
        parent = self.parents.get(chunk.parent_id)
        if not parent:
            return ""
        start = max(0, chunk.parent_position - radius)
        end = min(len(parent.segments), chunk.parent_position + radius + 1)
        return "\n\n".join(segment for segment in parent.segments[start:end] if segment != chunk.text)

    def to_dict(self) -> dict:
        return {
            "backend": "sentence-transformers",
            "model_name": self.model_name,
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "vectors": self.vectors.tolist(),
            "parents": [asdict(parent) for parent in self.parents.values()],
            "corpus_fingerprint": self.corpus_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingIndex":
        return cls(
            chunks=[Chunk(**chunk) for chunk in data["chunks"]],
            vectors=np.asarray(data["vectors"], dtype=np.float32),
            model_name=data.get("model_name", DEFAULT_MODEL),
            parents={parent["id"]: ParentChunk(**parent) for parent in data.get("parents", [])},
            corpus_fingerprint=data.get("corpus_fingerprint"),
        )

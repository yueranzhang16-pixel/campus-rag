from __future__ import annotations

from typing import Any

from .embeddings import EmbeddingIndex
from .retrieval import SearchResult

DEFAULT_RERANKER = "BAAI/bge-reranker-base"


class RerankingRetriever:
    """Re-rank dense-retrieval candidates with a CPU-loaded cross-encoder."""

    def __init__(
        self,
        base_index: EmbeddingIndex,
        model_name: str = DEFAULT_RERANKER,
        reranker: Any | None = None,
        candidate_k: int = 10,
        local_files_only: bool = True,
    ):
        self.base_index = base_index
        self.model_name = model_name
        self._reranker = reranker
        self.candidate_k = candidate_k
        self.local_files_only = local_files_only

    def _get_reranker(self) -> Any:
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "缺少 sentence-transformers。请先运行：python -m pip install sentence-transformers"
                ) from exc
            self._reranker = CrossEncoder(
                self.model_name, device="cpu", local_files_only=self.local_files_only
            )
        return self._reranker

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        candidates = self.base_index.search(question, top_k=self.candidate_k)
        if not candidates:
            return []
        scores = self._get_reranker().predict([[question, candidate.text] for candidate in candidates])
        rescored = [
            SearchResult(
                source=candidate.source,
                text=candidate.text,
                context=candidate.context,
                score=round(float(score), 4),
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(rescored, key=lambda result: result.score, reverse=True)[:top_k]

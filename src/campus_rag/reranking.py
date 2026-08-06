from __future__ import annotations

from typing import Any, Protocol

from .retrieval import SearchResult

DEFAULT_RERANKER = "BAAI/bge-reranker-base"


class Retriever(Protocol):
    """Minimal interface shared by dense and hybrid retrievers."""

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]: ...


class RerankingRetriever:
    """Re-rank candidates from any retriever with a CPU cross-encoder."""

    def __init__(
        self,
        base_retriever: Retriever,
        model_name: str = DEFAULT_RERANKER,
        reranker: Any | None = None,
        candidate_k: int = 10,
        local_files_only: bool = True,
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self.base_retriever = base_retriever
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
            try:
                self._reranker = CrossEncoder(
                    self.model_name, device="cpu", local_files_only=self.local_files_only
                )
            except OSError as exc:
                if self.local_files_only:
                    raise RuntimeError(
                        f"本地重排序模型 {self.model_name} 不完整。"
                        "请联网后在命令末尾添加 --allow-download 下载权重，再重试。"
                    ) from exc
                raise RuntimeError(f"无法加载或下载重排序模型 {self.model_name}。") from exc
        return self._reranker

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        candidates = self.base_retriever.search(question, top_k=self.candidate_k)
        if not candidates:
            return []
        scores = self._get_reranker().predict([[question, candidate.text] for candidate in candidates])
        rescored = [
            SearchResult(
                source=candidate.source,
                text=candidate.text,
                context=candidate.context,
                score=round(float(score), 4),
                parent_text=candidate.parent_text,
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(rescored, key=lambda result: result.score, reverse=True)[:top_k]

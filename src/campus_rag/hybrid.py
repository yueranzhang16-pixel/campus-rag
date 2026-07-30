from .embeddings import EmbeddingIndex
from .retrieval import SearchResult, TfidfIndex


class HybridRetriever:
    def __init__(self, dense: EmbeddingIndex, lexical: TfidfIndex):
        self.dense, self.lexical = dense, lexical

    @staticmethod
    def _content_adjustment(item: SearchResult, prefer_explanations: bool) -> float:
        """Prefer explanatory passages to code-only snippets for natural-language QA."""
        adjustment = 0.0
        if prefer_explanations and "```" in item.text:
            adjustment -= 0.05
        if "定义" in item.context:
            adjustment += 0.003
        if "注意事项" in item.context:
            adjustment += 0.002
        return adjustment

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        merged = {}
        code_question = any(term in question.lower() for term in ("代码", "函数", "实现", "c++", "c语言", "python"))
        prefer_explanations = not code_question
        candidate_k = max(top_k * 4, 12)
        for results in (
            self.dense.search(question, top_k=candidate_k),
            self.lexical.search(question, top_k=candidate_k),
        ):
            for rank, item in enumerate(results, 1):
                key = (item.source, item.text)
                merged[key] = (merged.get(key, (0, item))[0] + 1 / (60 + rank), item)
        ranked = sorted(
            merged.values(),
            reverse=True,
            key=lambda pair: pair[0] + self._content_adjustment(pair[1], prefer_explanations),
        )
        return [
            SearchResult(
                item.source,
                item.text,
                round(score + self._content_adjustment(item, prefer_explanations), 4),
                item.context,
            )
            for score, item in ranked[:top_k]
        ]

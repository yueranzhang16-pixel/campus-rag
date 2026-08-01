from .embeddings import EmbeddingIndex
from .retrieval import SearchResult, TfidfIndex


class HybridRetriever:
    LEXICAL_SCORE_WEIGHT = 0.10
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
        for backend, results in (
            ("dense", self.dense.search(question, top_k=candidate_k)),
            ("lexical", self.lexical.search(question, top_k=candidate_k)),
        ):
            for rank, item in enumerate(results, 1):
                # Context is part of a passage identity. Different code blocks can
                # have identical text (for example an empty fenced block); merging
                # them would incorrectly accumulate their RRF scores.
                key = (item.source, item.context, item.text)
                rrf_score = 1 / (60 + rank)
                # RRF rewards passages returned by both retrievers. Add a bounded
                # lexical-relevance signal so an exact terminology match is not
                # buried by two semantically similar but irrelevant passages.
                lexical_bonus = self.LEXICAL_SCORE_WEIGHT * item.score if backend == "lexical" else 0.0
                merged[key] = (merged.get(key, (0, item))[0] + rrf_score + lexical_bonus, item)
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
                item.parent_text,
            )
            for score, item in ranked[:top_k]
        ]

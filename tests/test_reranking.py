import unittest

from campus_rag.reranking import RerankingRetriever
from campus_rag.retrieval import SearchResult


class FakeBaseIndex:
    def search(self, _question, top_k):
        candidates = [
            SearchResult("first.md", "first passage", 0.9, "first"),
            SearchResult("correct.md", "correct passage", 0.8, "correct"),
        ]
        return candidates[:top_k]


class FakeParentBase:
    def search(self, _question, top_k):
        return [
            SearchResult(
                "tree.md", "child passage", 0.8, "Tree definition", "nearby parent context"
            )
        ][:top_k]


class FakeReranker:
    def predict(self, _pairs):
        if len(_pairs) == 1:
            return [0.95]
        return [0.1, 0.95]


class RerankingRetrieverTests(unittest.TestCase):
    def test_reranker_can_change_dense_retrieval_order(self):
        retriever = RerankingRetriever(FakeBaseIndex(), reranker=FakeReranker(), candidate_k=2)
        result = retriever.search("question", top_k=1)
        self.assertEqual(result[0].source, "correct.md")
        self.assertEqual(result[0].score, 0.95)

    def test_reranker_preserves_parent_context(self):
        retriever = RerankingRetriever(FakeParentBase(), reranker=FakeReranker(), candidate_k=1)
        result = retriever.search("question", top_k=1)
        self.assertEqual(result[0].parent_text, "nearby parent context")

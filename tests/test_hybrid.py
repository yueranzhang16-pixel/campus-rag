import unittest

from campus_rag.hybrid import HybridRetriever
from campus_rag.retrieval import Chunk, SearchResult, TfidfIndex


class FakeIndex:
    def __init__(self, results):
        self.results = results

    def search(self, question, top_k):
        return self.results[:top_k]


class HybridRetrieverTests(unittest.TestCase):
    def test_combines_dense_and_lexical_candidates(self):
        dense = TfidfIndex.build([Chunk("dense.md", "语义相关的内容")])
        lexical = TfidfIndex.build([Chunk("lexical.md", "线性探测再散列的增量序列")])
        retriever = HybridRetriever(dense, lexical)

        results = retriever.search("线性探测再散列的增量序列", top_k=2)

        self.assertEqual({item.source for item in results}, {"dense.md", "lexical.md"})

    def test_prefers_explanatory_passage_over_code(self):
        code = SearchResult("stack.md", "```C\nvoid Push() {}\n```", 0.9, "堆栈 顺序栈 测试")
        explanation = SearchResult("stack.md", "堆栈遵循后进先出原则。", 0.8, "堆栈 定义 注意事项")
        retriever = HybridRetriever(FakeIndex([code, explanation]), FakeIndex([code, explanation]))

        results = retriever.search("堆栈遵循什么操作顺序？", top_k=1)

        self.assertEqual(results[0].text, explanation.text)

    def test_exact_lexical_match_is_not_buried_by_dense_overlap(self):
        dense_only = SearchResult("dense.md", "```", 0.9, "代码示例一")
        dense_duplicate = SearchResult("dense.md", "```", 0.8, "代码示例二")
        exact_match = SearchResult("hash.md", "线性探测再散列的增量序列是 1，2，3。", 0.9, "冲突解决")
        retriever = HybridRetriever(
            FakeIndex([dense_only, dense_duplicate]),
            FakeIndex([exact_match, dense_only, dense_duplicate]),
        )

        results = retriever.search("线性探测再散列的增量序列是什么？", top_k=1)

        self.assertEqual(results[0].source, "hash.md")


if __name__ == "__main__":
    unittest.main()

import unittest
from campus_rag import cli

from campus_rag.retrieval import Chunk, SearchResult, TfidfIndex, split_markdown_chunks, split_markdown_document, tokenize


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = TfidfIndex.build(
            [
                Chunk("library.md", "本科生每次最多借阅 10 本图书，期限为 30 天。"),
                Chunk("course.md", "课程项目占成绩 40%，第 16 周前提交。"),
            ]
        )

    def test_chinese_tokenization_is_not_empty(self):
        self.assertGreater(len(tokenize("借书期限")), 0)

    def test_library_question_returns_library_source(self):
        result = self.index.search("图书借阅期限", top_k=1)
        self.assertEqual(result[0].source, "library.md")

    def test_course_question_returns_course_source(self):
        result = self.index.search("课程项目成绩", top_k=1)
        self.assertEqual(result[0].source, "course.md")

    def test_section_context_participates_in_retrieval(self):
        index = TfidfIndex.build(
            [
                Chunk("tree.md", "该操作会依次访问结点。", context="树 先序遍历"),
                Chunk("array.md", "该操作会依次访问元素。", context="数组 二维存储"),
            ]
        )
        result = index.search("树的先序遍历", top_k=1)
        self.assertEqual(result[0].source, "tree.md")

    def test_markdown_chunks_keep_parent_and_child_heading_context(self):
        chunks = split_markdown_chunks(
            "tree.md",
            "## 树的基本操作\n### 遍历\n#### 先序\n按根到孩子的顺序访问。",
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("树的基本操作", chunks[0].context)
        self.assertIn("遍历", chunks[0].context)
        self.assertIn("先序", chunks[0].context)

    def test_parent_child_retrieval_expands_a_heading_with_nearby_details(self):
        chunks, parents = split_markdown_document(
            "tree.md",
            "## B树与B+树\n### B+树\n差异：\n\n1. 数据只存于叶子结点。\n\n2. 叶子结点按关键字链接。",
        )
        index = TfidfIndex.build(chunks, parents)

        result = index.search("B树和B+树有什么差异？", top_k=1)[0]

        self.assertEqual(result.text, "差异：")
        self.assertIn("数据只存于叶子结点", result.parent_text)
        self.assertIn("叶子结点按关键字链接", result.parent_text)

    def test_parent_metadata_survives_index_serialization(self):
        chunks, parents = split_markdown_document("tree.md", "## 树\n定义：\n\n树由结点组成。")
        restored = TfidfIndex.from_dict(TfidfIndex.build(chunks, parents).to_dict())

        self.assertTrue(restored.parents)
        self.assertEqual(restored.search("树的定义", top_k=1)[0].parent_text, "树由结点组成。")

    def test_corpus_fingerprint_survives_index_serialization(self):
        restored = TfidfIndex.from_dict(TfidfIndex.build(self.index.chunks, corpus_fingerprint="abc123").to_dict())
        self.assertEqual(restored.corpus_fingerprint, "abc123")

    def test_eval_reports_per_case_hits(self):
        report = cli.evaluate(
            self.index,
            [{"question": "图书借阅期限", "expected_source": "library.md"}],
            top_k=1,
        )
        self.assertEqual(report["score"], 1.0)
        self.assertTrue(report["cases"][0]["hit"])

    def test_eval_accepts_multiple_valid_sources(self):
        report = cli.evaluate(
            self.index,
            [{"question": "课程项目成绩", "expected_sources": ["other.md", "course.md"]}],
            top_k=1,
        )
        self.assertEqual(report["score"], 1.0)

    def test_eval_reports_metrics_by_category(self):
        report = cli.evaluate(
            self.index,
            [
                {"question": "图书借阅期限", "expected_source": "library.md", "category": "definition"},
                {"question": "课程项目成绩", "expected_source": "course.md", "category": "definition"},
            ],
            top_k=1,
        )
        self.assertEqual(report["category_metrics"]["definition"], {"hits": 2, "total": 2, "score": 1.0})

    def test_abstention_eval_reports_precision_and_recall(self):
        class StubRetriever:
            def search(self, question, top_k):
                score = 0.02 if question == "unknown" else 0.08
                return [SearchResult("source.md", "passage", score)][:top_k]

        report = cli.evaluate_abstention(
            StubRetriever(),
            [
                {"question": "known", "expect_refusal": False},
                {"question": "unknown", "expect_refusal": True},
            ],
            top_k=1,
        )
        self.assertEqual(report["score"], 1.0)
        self.assertEqual(report["refusal_precision"], 1.0)
        self.assertEqual(report["refusal_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()

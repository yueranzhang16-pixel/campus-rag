import unittest
from campus_rag import cli

from campus_rag.retrieval import Chunk, TfidfIndex, split_markdown_chunks, tokenize


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


if __name__ == "__main__":
    unittest.main()

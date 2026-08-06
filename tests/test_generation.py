import unittest

from campus_rag.generation import (
    INSUFFICIENT_EVIDENCE_RESPONSE,
    DeepSeekGenerator,
    assess_evidence,
    build_messages,
    check_answer,
)
from campus_rag.retrieval import SearchResult


class GenerationTests(unittest.TestCase):
    def test_prompt_includes_evidence_and_source(self):
        messages = build_messages(
            "什么是顺序表？",
            [SearchResult("线性表.md", "顺序表使用连续存储单元。", 0.9, "线性表 顺序表")],
        )
        self.assertIn("线性表.md", messages[1]["content"])

    def test_prompt_includes_parent_context_when_available(self):
        messages = build_messages(
            "B树和B+树的差异是什么？",
            [SearchResult("数据结构.md", "差异：", 0.5, "B+树", "1. 数据只存于叶子结点。")],
        )
        self.assertIn("相邻父级上下文", messages[1]["content"])
        self.assertIn("数据只存于叶子结点", messages[1]["content"])

    def test_answer_check_requires_terms_and_citation(self):
        check = check_answer("地址连续的存储单元。[线性表.md]", ["地址连续"], ["线性表.md"])
        self.assertTrue(check["terms_pass"] and check["citation_pass"])

    def test_answer_check_accepts_citation_with_section_context(self):
        check = check_answer(
            "空串长度为 0。[来源：串.md｜章节：串 空格串与空串]",
            ["空串"],
            ["串.md"],
        )
        self.assertTrue(check["terms_pass"] and check["citation_pass"])

    def test_answer_check_accepts_citation_with_anchor(self):
        check = check_answer("top 为 -1 表示空栈。[堆栈.md/#顺序栈注意事项]", ["空栈"], ["堆栈.md"])
        self.assertTrue(check["terms_pass"] and check["citation_pass"])

    def test_low_confidence_evidence_returns_refusal_without_an_api_call(self):
        generator = DeepSeekGenerator(api_key="not-used")
        generator.complete = lambda *_args, **_kwargs: self.fail("API should not be called")

        answer = generator.answer_with_usage(
            "光合作用的公式是什么？", [SearchResult("串.md", "无关片段", 0.03)]
        )

        self.assertEqual(answer.content, INSUFFICIENT_EVIDENCE_RESPONSE)
        self.assertEqual(answer.usage, {})

    def test_evidence_gate_keeps_score_at_or_above_threshold(self):
        assessment = assess_evidence([SearchResult("队列.md", "队列片段", 0.06)])
        self.assertTrue(assessment.sufficient)

    def test_evidence_gate_uses_the_first_ranked_result(self):
        assessment = assess_evidence(
            [SearchResult("first.md", "低置信片段", 0.02), SearchResult("later.md", "后续片段", 0.9)]
        )
        self.assertFalse(assessment.sufficient)

    def test_evidence_gate_accepts_a_low_score_with_an_exact_english_anchor(self):
        assessment = assess_evidence(
            [SearchResult("graph.md", "Dijkstra solves single-source shortest paths", 0.03)],
            "What does Dijkstra do?",
        )
        self.assertTrue(assessment.sufficient)
        self.assertEqual(assessment.reason, "exact_english_anchor")

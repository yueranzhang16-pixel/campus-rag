import unittest

from campus_rag.generation import GeneratedAnswer
from campus_rag.judge import GroundedAnswerJudge, build_judge_messages, parse_judgement
from campus_rag.retrieval import SearchResult


class JudgeTests(unittest.TestCase):
    def test_prompt_includes_evidence_and_length_bias_guardrail(self):
        messages = build_judge_messages(
            "什么是顺序表？",
            "顺序表使用连续地址。 [线性表.md]",
            [SearchResult("线性表.md", "顺序表使用连续地址。", 0.9, "定义")],
        )
        self.assertIn("不要奖励篇幅", messages[0]["content"])
        self.assertIn("线性表.md", messages[1]["content"])

    def test_parser_normalizes_weighted_scores(self):
        result = parse_judgement(
            '{"criteria":['
            '{"name":"faithfulness","score":5,"justification":"证据支持","improvement":"无"},'
            '{"name":"relevance","score":4,"justification":"回答问题","improvement":"更简洁"},'
            '{"name":"citation_support","score":5,"justification":"引用正确","improvement":"无"},'
            '{"name":"abstention","score":4,"justification":"处理恰当","improvement":"无"}'
            '],"confidence":1,"summary":"整体可靠"}'
        )
        self.assertEqual(result["weighted_score"], 4.65)
        self.assertEqual(result["confidence"], 0.99)
        self.assertEqual(len(result["criteria"]), 4)

    def test_parser_rejects_missing_criterion(self):
        with self.assertRaises(ValueError):
            parse_judgement('{"criteria": [{"name": "faithfulness", "score": 5}]}')

    def test_judge_adds_model_version_and_usage(self):
        class FakeClient:
            model = "judge-test"

            def complete(self, messages, temperature):
                return GeneratedAnswer(
                    '{"criteria":['
                    '{"name":"faithfulness","score":5,"justification":"支持","improvement":"无"},'
                    '{"name":"relevance","score":5,"justification":"相关","improvement":"无"},'
                    '{"name":"citation_support","score":5,"justification":"引用","improvement":"无"},'
                    '{"name":"abstention","score":5,"justification":"恰当","improvement":"无"}'
                    '],"confidence":0.8,"summary":"通过"}',
                    {"prompt_tokens": 10},
                )

        result = GroundedAnswerJudge(FakeClient()).evaluate("问题", "回答", [])
        self.assertEqual(result["judge_model"], "judge-test")
        self.assertEqual(result["judge_usage"], {"prompt_tokens": 10})
        self.assertEqual(result["weighted_score"], 5.0)


if __name__ == "__main__":
    unittest.main()

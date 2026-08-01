import unittest

from campus_rag.feedback_report import build_feedback_report


class FeedbackReportTests(unittest.TestCase):
    def test_report_joins_down_feedback_to_answer_and_groups_reasons(self):
        report = build_feedback_report(
            [{"id": "answer-1", "question": "二叉树有哪些种类？", "answer": "资料不足", "sources": ["数据结构.md"]}],
            [
                {"answer_id": "answer-1", "rating": "down", "reason": "missing_knowledge", "note": "应补充分类", "timestamp": "2026-08-01T00:00:00+00:00"},
                {"answer_id": "answer-2", "rating": "up", "timestamp": "2026-08-01T00:01:00+00:00"},
            ],
        )
        self.assertEqual(report["ratings"], {"up": 1, "down": 1})
        self.assertEqual(report["reasons"], {"missing_knowledge": 1})
        self.assertEqual(report["review_queue"][0]["question"], "二叉树有哪些种类？")
        self.assertEqual(report["review_queue"][0]["reason_label"], "资料缺失")


if __name__ == "__main__":
    unittest.main()

import unittest

from campus_rag.trace_report import build_trace_report


class TraceReportTests(unittest.TestCase):
    def test_report_calculates_latency_and_marks_legacy_records(self):
        report = build_trace_report(
            [
                {"id": "old", "question": "旧问题", "latency_ms": 10, "sources": ["a.md"]},
                {
                    "trace_id": "new",
                    "question": "新问题",
                    "latency_ms": 110,
                    "sources": ["a.md", "b.md"],
                    "retrieval": [{"rank": 1, "score": 0.3}],
                    "trace": {"model": "test-model"},
                },
            ]
        )
        self.assertEqual(report["total_traces"], 2)
        self.assertEqual(report["latency_ms"]["average"], 60)
        self.assertEqual(report["latency_ms"]["p95"], 110)
        self.assertEqual(report["slow_traces"][0]["trace_id"], "new")
        self.assertEqual(report["incomplete_trace_count"], 1)


if __name__ == "__main__":
    unittest.main()

import unittest

from campus_rag.quality_gate import evaluate_quality_gate


class QualityGateTests(unittest.TestCase):
    def test_quality_gate_passes_when_metrics_and_indexes_meet_thresholds(self):
        report = evaluate_quality_gate(
            {"score": 1.0}, {"score": 1.0}, {"embedding": "fresh", "lexical": "fresh"}
        )
        self.assertTrue(report["passed"])

    def test_quality_gate_rejects_stale_indexes(self):
        report = evaluate_quality_gate(
            {"score": 1.0}, {"score": 1.0}, {"embedding": "stale", "lexical": "fresh"}
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["embedding_index_fresh"]["passed"])

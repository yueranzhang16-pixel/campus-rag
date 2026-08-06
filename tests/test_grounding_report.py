import unittest

from campus_rag.grounding_report import build_grounding_report


class GroundingReportTests(unittest.TestCase):
    def test_report_counts_refusals_and_invalid_citations(self):
        report = build_grounding_report(
            [
                {
                    "id": "valid",
                    "trace": {
                        "evidence_gate": {"sufficient": True},
                        "citation": {"citation_required": True, "citation_valid": True},
                    },
                },
                {
                    "id": "invalid",
                    "question": "question",
                    "trace": {
                        "evidence_gate": {"sufficient": False},
                        "citation": {
                            "citation_required": False,
                            "citation_valid": False,
                            "cited_sources": ["wrong.md"],
                            "unsupported_sources": ["wrong.md"],
                        },
                    },
                },
            ]
        )
        self.assertEqual(report["refused_count"], 1)
        self.assertEqual(report["citation_valid_rate"], 1.0)
        self.assertEqual(report["invalid_citation_examples"][0]["trace_id"], "invalid")

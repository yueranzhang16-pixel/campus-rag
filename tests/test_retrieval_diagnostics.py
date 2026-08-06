import unittest

from campus_rag.cli import benchmark_retrieval
from campus_rag.retrieval import SearchResult
from campus_rag.retrieval_diagnostics import build_retrieval_diagnostic


class FakeBackend:
    def __init__(self, results):
        self.results = results

    def search(self, _question, top_k):
        return self.results[:top_k]


class FakeHybrid:
    def __init__(self):
        shared = SearchResult("shared.md", "shared", 0.8, "shared")
        self.dense = FakeBackend([shared, SearchResult("dense.md", "dense", 0.7)])
        self.lexical = FakeBackend([shared, SearchResult("lexical.md", "lexical", 0.6)])

    def search(self, _question, top_k):
        return self.dense.search(_question, top_k)


class RetrievalDiagnosticsTests(unittest.TestCase):
    def test_diagnostic_reports_backend_overlap(self):
        report = build_retrieval_diagnostic(FakeHybrid(), "question", top_k=2)
        self.assertEqual(report["overlap"]["dense_lexical_shared_count"], 1)
        self.assertEqual(len(report["hybrid"]), 2)

    def test_benchmark_reports_warm_query_count(self):
        report = benchmark_retrieval(
            FakeBackend([SearchResult("source.md", "text", 0.8)]),
            [{"question": "one"}, {"question": "two"}],
            top_k=1,
            repeats=2,
        )
        self.assertEqual(report["queries"], 4)
        self.assertGreaterEqual(report["latency_ms"]["max"], 0)

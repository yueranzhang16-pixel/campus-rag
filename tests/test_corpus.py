import unittest
from pathlib import Path

from campus_rag.corpus import build_corpus_manifest


class CorpusTests(unittest.TestCase):
    def test_manifest_is_stable_for_current_corpus(self):
        docs_dir = Path(__file__).resolve().parents[1] / "data" / "docs"
        first = build_corpus_manifest(docs_dir)
        second = build_corpus_manifest(docs_dir)
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertGreater(first["file_count"], 0)

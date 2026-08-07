import json
import unittest
from pathlib import Path

from campus_rag.corpus import build_corpus_manifest
from campus_rag.index_status import get_index_freshness


class IndexStatusTests(unittest.TestCase):
    def test_missing_index_is_reported(self):
        docs_dir = Path(__file__).resolve().parents[1] / "data" / "docs"
        self.assertEqual(get_index_freshness(Path("missing-index.json"), docs_dir), "missing")

    def test_current_manifest_matches_a_temporary_serialized_fingerprint(self):
        docs_dir = Path(__file__).resolve().parents[1] / "data" / "docs"
        fixture = Path(__file__).resolve().parents[1] / "logs" / "test_index_fingerprint.json"
        fixture.parent.mkdir(exist_ok=True)
        fixture.write_text(json.dumps({"corpus_fingerprint": build_corpus_manifest(docs_dir)["fingerprint"]}), encoding="utf-8")
        self.assertEqual(get_index_freshness(fixture, docs_dir), "fresh")

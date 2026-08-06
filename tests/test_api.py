import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from campus_rag.api import CampusRagService, create_app
from campus_rag.generation import INSUFFICIENT_EVIDENCE_RESPONSE
from campus_rag.retrieval import SearchResult


class FakeRetriever:
    def search(self, question, top_k):
        return [SearchResult("线性表.md", "顺序表使用连续地址存储。", 0.9, "线性表 顺序表")][:top_k]


class FakeGenerator:
    def answer(self, question, evidence):
        return "顺序表使用连续地址存储。[线性表.md]"


class LowConfidenceRetriever:
    def search(self, question, top_k):
        return [SearchResult("线性表.md", "无关片段", 0.02, "线性表")][:top_k]


class FakeService(CampusRagService):
    @property
    def retriever(self):
        return FakeRetriever()

    @property
    def generator(self):
        return FakeGenerator()


class LowConfidenceService(FakeService):
    @property
    def retriever(self):
        return LowConfidenceRetriever()


class ApiTests(unittest.TestCase):
    def setUp(self):
        history_path = Path(__file__).resolve().parents[1] / "logs" / "test_answer_history.jsonl"
        feedback_path = Path(__file__).resolve().parents[1] / "logs" / "test_feedback.jsonl"
        service = FakeService(history_path=history_path, feedback_path=feedback_path)
        self.client = TestClient(create_app(service))

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_ready_reports_indexes_and_does_not_expose_the_api_key(self):
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["status"], {"ok", "degraded"})
        self.assertIn("embedding_index_exists", response.json())
        self.assertIn("api_key_configured", response.json())
        self.assertNotIn("api_key", response.json())

    def test_index_signature_marks_missing_index_without_breaking_trace_setup(self):
        signature = CampusRagService._index_signature(Path("missing-index.json"))
        self.assertTrue(signature["missing"])
        self.assertIsNone(signature["sha256"])

    def test_injected_retriever_can_save_history_without_local_index_artifacts(self):
        root = Path(__file__).resolve().parents[1] / "logs"
        service = FakeService(
            embedding_index=root / "missing-embedding.json",
            lexical_index=root / "missing-lexical.json",
            history_path=root / "missing_index_history.jsonl",
            feedback_path=root / "missing_index_feedback.jsonl",
        )
        client = TestClient(create_app(service))
        answer = client.post("/answer", json={"question": "什么是顺序表？"}).json()
        self.assertTrue(answer["answer_id"])
        history = client.get("/history").json()["records"]
        self.assertTrue(history[0]["trace"]["index_versions"]["embedding"]["missing"])

    def test_root_returns_chat_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("校园知识库问答", response.text)
        self.assertIn("每行一个问题", response.text)

    def test_retrieve_returns_evidence(self):
        response = self.client.post("/retrieve", json={"question": "什么是顺序表？"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["evidence"][0]["source"], "线性表.md")

    def test_answer_returns_answer_and_evidence(self):
        response = self.client.post("/answer", json={"question": "什么是顺序表？"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("线性表.md", response.json()["answer"])
        self.assertEqual(response.json()["evidence"][0]["source"], "线性表.md")
        self.assertTrue(response.json()["answer_id"])
        self.assertEqual(response.json()["trace_id"], response.json()["answer_id"])
        self.assertTrue(response.json()["grounding"]["citation_valid"])

    def test_history_keeps_answer_metadata_locally(self):
        self.client.post("/answer", json={"question": "什么是顺序表？"})
        response = self.client.get("/history")
        record = response.json()["records"][0]
        self.assertEqual(record["question"], "什么是顺序表？")
        self.assertEqual(record["sources"], ["线性表.md"])
        self.assertIn("latency_ms", record)
        self.assertEqual(record["retrieval"][0]["score"], 0.9)
        self.assertEqual(record["trace"]["retriever"], "hybrid_rrf")
        self.assertIn("embedding", record["trace"]["index_versions"])
        self.assertIn("citation", record["trace"])

    def test_feedback_is_linked_to_answer_id(self):
        answer = self.client.post("/answer", json={"question": "什么是顺序表？"}).json()
        response = self.client.post("/feedback", json={"answer_id": answer["answer_id"], "rating": "up"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["record"]["rating"], "up")
        self.assertEqual(response.json()["record"]["answer_id"], answer["answer_id"])

    def test_down_feedback_requires_a_reason_and_keeps_the_note(self):
        answer = self.client.post("/answer", json={"question": "什么是顺序表？"}).json()
        missing_reason = self.client.post("/feedback", json={"answer_id": answer["answer_id"], "rating": "down"})
        self.assertEqual(missing_reason.status_code, 422)
        response = self.client.post(
            "/feedback",
            json={
                "answer_id": answer["answer_id"],
                "rating": "down",
                "reason": "missing_knowledge",
                "note": "资料没有覆盖这个概念。",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["record"]["reason"], "missing_knowledge")
        self.assertEqual(response.json()["record"]["note"], "资料没有覆盖这个概念。")

    def test_rejects_empty_question(self):
        response = self.client.post("/retrieve", json={"question": ""})
        self.assertEqual(response.status_code, 422)

    def test_answer_refuses_when_evidence_score_is_too_low(self):
        service = LowConfidenceService(history_path=None, feedback_path=None)
        response = TestClient(create_app(service)).post("/answer", json={"question": "光合作用是什么？"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], INSUFFICIENT_EVIDENCE_RESPONSE)


if __name__ == "__main__":
    unittest.main()

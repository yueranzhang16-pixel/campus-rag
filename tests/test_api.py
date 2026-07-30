import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from campus_rag.api import CampusRagService, create_app
from campus_rag.retrieval import SearchResult


class FakeRetriever:
    def search(self, question, top_k):
        return [SearchResult("线性表.md", "顺序表使用连续地址存储。", 0.9, "线性表 顺序表")][:top_k]


class FakeGenerator:
    def answer(self, question, evidence):
        return "顺序表使用连续地址存储。[线性表.md]"


class FakeService(CampusRagService):
    @property
    def retriever(self):
        return FakeRetriever()

    @property
    def generator(self):
        return FakeGenerator()


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

    def test_history_keeps_answer_metadata_locally(self):
        self.client.post("/answer", json={"question": "什么是顺序表？"})
        response = self.client.get("/history")
        record = response.json()["records"][0]
        self.assertEqual(record["question"], "什么是顺序表？")
        self.assertEqual(record["sources"], ["线性表.md"])
        self.assertIn("latency_ms", record)

    def test_feedback_is_linked_to_answer_id(self):
        answer = self.client.post("/answer", json={"question": "什么是顺序表？"}).json()
        response = self.client.post("/feedback", json={"answer_id": answer["answer_id"], "rating": "up"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["record"]["rating"], "up")
        self.assertEqual(response.json()["record"]["answer_id"], answer["answer_id"])

    def test_rejects_empty_question(self):
        response = self.client.post("/retrieve", json={"question": ""})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()

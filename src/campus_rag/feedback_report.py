from __future__ import annotations

from collections import Counter


REASON_LABELS = {
    "missing_knowledge": "资料缺失",
    "irrelevant": "答非所问",
    "incorrect": "内容错误",
    "unclear": "表达不清",
}


def build_feedback_report(answer_records: list[dict], feedback_records: list[dict]) -> dict:
    """Join local feedback to answers and expose a review queue for RAG iteration."""
    answers_by_id = {record.get("id"): record for record in answer_records if record.get("id")}
    ratings = Counter(record.get("rating", "unknown") for record in feedback_records)
    reasons = Counter(
        record["reason"]
        for record in feedback_records
        if record.get("rating") == "down" and record.get("reason")
    )
    review_queue = []
    for feedback in reversed(feedback_records):
        if feedback.get("rating") != "down":
            continue
        answer = answers_by_id.get(feedback.get("answer_id"), {})
        review_queue.append(
            {
                "answer_id": feedback.get("answer_id"),
                "timestamp": feedback.get("timestamp"),
                "reason": feedback.get("reason") or "unspecified",
                "reason_label": REASON_LABELS.get(feedback.get("reason"), "未说明"),
                "note": feedback.get("note", ""),
                "question": answer.get("question", "（原问答记录未找到）"),
                "answer": answer.get("answer", ""),
                "sources": answer.get("sources", []),
            }
        )
    return {
        "total_feedback": len(feedback_records),
        "ratings": {"up": ratings["up"], "down": ratings["down"]},
        "reasons": dict(reasons),
        "review_queue": review_queue,
    }

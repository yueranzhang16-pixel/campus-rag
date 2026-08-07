from __future__ import annotations


def evaluate_quality_gate(
    retrieval_report: dict,
    abstention_report: dict,
    index_freshness: dict[str, str],
    min_recall: float = 1.0,
    min_abstention_accuracy: float = 1.0,
) -> dict:
    """Apply explicit release thresholds to existing offline evaluation reports."""
    checks = {
        "retrieval_recall": {
            "actual": retrieval_report.get("score", 0.0),
            "threshold": min_recall,
        },
        "abstention_accuracy": {
            "actual": abstention_report.get("score", 0.0),
            "threshold": min_abstention_accuracy,
        },
        "embedding_index_fresh": {"actual": index_freshness.get("embedding"), "threshold": "fresh"},
        "lexical_index_fresh": {"actual": index_freshness.get("lexical"), "threshold": "fresh"},
    }
    checks["retrieval_recall"]["passed"] = checks["retrieval_recall"]["actual"] >= min_recall
    checks["abstention_accuracy"]["passed"] = checks["abstention_accuracy"]["actual"] >= min_abstention_accuracy
    checks["embedding_index_fresh"]["passed"] = checks["embedding_index_fresh"]["actual"] == "fresh"
    checks["lexical_index_fresh"]["passed"] = checks["lexical_index_fresh"]["actual"] == "fresh"
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}

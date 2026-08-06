from __future__ import annotations


def build_grounding_report(records: list[dict], limit: int = 20) -> dict:
    """Summarize evidence-gate and citation validation results from local traces."""
    evaluated = [record for record in records if record.get("trace", {}).get("citation")]
    citations = [record["trace"]["citation"] for record in evaluated]
    required = [citation for citation in citations if citation.get("citation_required")]
    invalid = [record for record in evaluated if not record["trace"]["citation"].get("citation_valid")]
    refused = [
        record
        for record in evaluated
        if not record.get("trace", {}).get("evidence_gate", {}).get("sufficient", True)
    ]
    return {
        "total_traces": len(records),
        "evaluated_traces": len(evaluated),
        "refused_count": len(refused),
        "citation_required_count": len(required),
        "citation_valid_rate": round(
            sum(citation.get("citation_valid", False) for citation in required) / len(required), 4
        ) if required else 0.0,
        "invalid_citation_examples": [
            {
                "trace_id": record.get("trace_id", record.get("id")),
                "question": record.get("question", ""),
                "cited_sources": record["trace"]["citation"].get("cited_sources", []),
                "unsupported_sources": record["trace"]["citation"].get("unsupported_sources", []),
            }
            for record in invalid[:limit]
        ],
    }

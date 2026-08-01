from __future__ import annotations

from collections import Counter
from statistics import mean


def build_trace_report(records: list[dict], limit: int = 10) -> dict:
    """Summarize local answer traces without exposing questions or answers externally."""
    latencies = [record["latency_ms"] for record in records if isinstance(record.get("latency_ms"), (int, float))]
    ordered_latencies = sorted(latencies)
    p95_index = max(0, round((len(ordered_latencies) - 1) * 0.95)) if ordered_latencies else 0
    source_counts = Counter(source for record in records for source in record.get("sources", []))
    slow = sorted(records, key=lambda record: record.get("latency_ms", 0), reverse=True)[:limit]
    incomplete = [record for record in records if not record.get("retrieval") or not record.get("trace")]
    return {
        "total_traces": len(records),
        "latency_ms": {
            "average": round(mean(latencies), 1) if latencies else 0,
            "p95": round(ordered_latencies[p95_index], 1) if ordered_latencies else 0,
        },
        "top_sources": [{"source": source, "count": count} for source, count in source_counts.most_common()],
        "slow_traces": [
            {
                "trace_id": record.get("trace_id", record.get("id")),
                "question": record.get("question", ""),
                "latency_ms": record.get("latency_ms", 0),
                "sources": record.get("sources", []),
                "model": record.get("trace", {}).get("model"),
            }
            for record in slow
        ],
        "incomplete_trace_count": len(incomplete),
    }

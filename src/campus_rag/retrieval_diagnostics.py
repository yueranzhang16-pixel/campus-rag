from __future__ import annotations

from .hybrid import HybridRetriever
from .retrieval import SearchResult


def _serialize(results: list[SearchResult]) -> list[dict]:
    return [
        {
            "rank": rank,
            "source": item.source,
            "context": item.context,
            "score": item.score,
            "text": item.text,
        }
        for rank, item in enumerate(results, 1)
    ]


def build_retrieval_diagnostic(retriever: HybridRetriever, question: str, top_k: int = 3) -> dict:
    """Expose backend differences for a single query without calling an LLM."""
    dense = retriever.dense.search(question, top_k=top_k)
    lexical = retriever.lexical.search(question, top_k=top_k)
    hybrid = retriever.search(question, top_k=top_k)
    dense_keys = {(item.source, item.context, item.text) for item in dense}
    lexical_keys = {(item.source, item.context, item.text) for item in lexical}
    return {
        "question": question,
        "top_k": top_k,
        "overlap": {
            "dense_lexical_shared_count": len(dense_keys & lexical_keys),
            "dense_lexical_union_count": len(dense_keys | lexical_keys),
        },
        "dense": _serialize(dense),
        "lexical": _serialize(lexical),
        "hybrid": _serialize(hybrid),
    }

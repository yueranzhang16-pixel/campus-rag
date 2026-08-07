# Changelog

## v0.3.0 — Day 21 to Day 30

- Added deterministic corpus fingerprints to lexical and embedding indexes.
- Added `/ready` freshness states and the `index-status` command to detect stale indexes after document updates.
- Added document linting for empty files, duplicate source names, oversized chunks, and instruction-like source content.
- Added a prompt safety boundary that treats retrieved material as untrusted data.
- Added a release `quality-gate` that requires fresh indexes, retrieval recall, and abstention accuracy to meet configured thresholds.
- Rebuilt both local indexes and passed the Day 30 release gate: fresh indexes, difficult retrieval 16/16, abstention 7/7.

## v0.2.0 — Day 11 to Day 20

- Added evidence-gated refusal before calling DeepSeek, plus an offline abstention evaluation set.
- Added citation validation to API responses and local answer traces.
- Added grounding, retrieval-diagnostic, and steady-state benchmark commands.
- Added 16 difficult retrieval cases with category-level metrics.
- Added `/ready`, a safe PowerShell configuration template, and GitHub Actions regression testing.
- Expanded architecture, reproducibility, and limitation documentation.

## v0.1.0 — Day 1 to Day 10

- Built a Chinese course RAG service with TF-IDF + BGE hybrid retrieval, RRF fusion, parent-child context, DeepSeek generation, FastAPI UI, local feedback, tracing, and LLM-as-judge evaluation.

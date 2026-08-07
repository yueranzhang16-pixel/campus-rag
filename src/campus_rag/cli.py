from __future__ import annotations

import argparse
import json
import time
from statistics import median
from pathlib import Path

from .corpus import build_corpus_manifest
from .document_quality import lint_documents
from .embeddings import DEFAULT_MODEL, EmbeddingIndex
from .feedback_report import REASON_LABELS, build_feedback_report
from .generation import DEFAULT_MODEL as DEFAULT_DEEPSEEK_MODEL, DeepSeekGenerator, assess_evidence, check_answer
from .history import AnswerHistory, FeedbackHistory
from .index_status import get_index_freshness
from .grounding_report import build_grounding_report
from .hybrid import HybridRetriever
from .judge import GroundedAnswerJudge
from .quality_gate import evaluate_quality_gate
from .reranking import DEFAULT_RERANKER, RerankingRetriever
from .retrieval_diagnostics import build_retrieval_diagnostic
from .retrieval import TfidfIndex, load_chunks, load_parent_child_corpus
from .trace_report import build_trace_report


def load_index(path: Path) -> TfidfIndex:
    return TfidfIndex.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_embedding_index(path: Path) -> EmbeddingIndex:
    return EmbeddingIndex.from_dict(json.loads(path.read_text(encoding="utf-8")))


def command_index(args: argparse.Namespace) -> None:
    docs_dir = Path(args.docs)
    chunks, parents = load_parent_child_corpus(docs_dir)
    index = TfidfIndex.build(chunks, parents, build_corpus_manifest(docs_dir)["fingerprint"])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed {len(index.chunks)} chunks -> {destination}")


def command_docs_lint(args: argparse.Namespace) -> None:
    report = lint_documents(Path(args.docs), args.max_chunk_chars)
    print(f"documents={report['documents']} chunks={report['chunks']} warnings={report['warning_count']}")
    for key in ("empty_documents", "duplicate_filenames", "oversized_chunks", "instruction_like_lines"):
        if report[key]:
            print(f"{key}: {len(report[key])}")
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report saved -> {destination}")
    if args.fail_on_warning and report["warning_count"]:
        raise RuntimeError("Document lint found warnings")


def index_freshness_summary(args: argparse.Namespace) -> dict[str, str]:
    docs_dir = Path(args.docs)
    return {
        "embedding": get_index_freshness(Path(args.index), docs_dir),
        "lexical": get_index_freshness(Path(args.lexical_index), docs_dir),
    }


def command_index_status(args: argparse.Namespace) -> None:
    freshness = index_freshness_summary(args)
    print(f"embedding={freshness['embedding']} lexical={freshness['lexical']}")
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"index_freshness": freshness}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report saved -> {destination}")
    if args.require_fresh and any(value != "fresh" for value in freshness.values()):
        raise RuntimeError("Indexes are not fresh; rebuild both indexes before release")


def command_quality_gate(args: argparse.Namespace) -> None:
    retrieval_report = json.loads(Path(args.retrieval_report).read_text(encoding="utf-8"))
    abstention_report = json.loads(Path(args.abstention_report).read_text(encoding="utf-8"))
    report = evaluate_quality_gate(
        retrieval_report,
        abstention_report,
        index_freshness_summary(args),
        args.min_recall,
        args.min_abstention_accuracy,
    )
    print(f"quality gate={'PASS' if report['passed'] else 'FAIL'}")
    for name, check in report["checks"].items():
        print(f"- {name}: actual={check['actual']} threshold={check['threshold']} passed={check['passed']}")
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError("Quality gate failed")


def command_query(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    results = load_index(Path(args.index)).search(args.question, args.top_k)
    latency_ms = (time.perf_counter() - started) * 1000
    if not results or results[0].score == 0:
        print("未找到足够相关的证据；请补充资料或换一种问法。")
    for number, result in enumerate(results, 1):
        context = f" section={result.context}" if result.context else ""
        print(f"\n[{number}] source={result.source}{context} score={result.score}\n{result.text}")
    print(f"\nlatency_ms={latency_ms:.1f}")


def print_results(results: list, latency_ms: float) -> None:
    if not results or results[0].score == 0:
        print("未找到足够相关的证据；请补充资料或换一种问法。")
    for number, result in enumerate(results, 1):
        context = f" section={result.context}" if result.context else ""
        print(f"\n[{number}] source={result.source}{context} score={result.score}\n{result.text}")
    print(f"\nlatency_ms={latency_ms:.1f}")


def command_embedding_index(args: argparse.Namespace) -> None:
    docs_dir = Path(args.docs)
    chunks, parents = load_parent_child_corpus(docs_dir)
    index = EmbeddingIndex.build(
        chunks,
        model_name=args.model,
        parents=parents,
        corpus_fingerprint=build_corpus_manifest(docs_dir)["fingerprint"],
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(index.to_dict(), ensure_ascii=False), encoding="utf-8")
    print(f"Embedded {len(index.chunks)} chunks with {index.model_name} -> {destination}")


def command_embedding_query(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    results = load_embedding_index(Path(args.index)).search(args.question, args.top_k)
    print_results(results, (time.perf_counter() - started) * 1000)


def load_hybrid_retriever(embedding_index: str, lexical_index: str) -> HybridRetriever:
    return HybridRetriever(
        load_embedding_index(Path(embedding_index)),
        load_index(Path(lexical_index)),
    )


def command_hybrid_query(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    results = load_hybrid_retriever(args.index, args.lexical_index).search(args.question, args.top_k)
    print_results(results, (time.perf_counter() - started) * 1000)


def command_diagnose_query(args: argparse.Namespace) -> None:
    report = build_retrieval_diagnostic(
        load_hybrid_retriever(args.index, args.lexical_index), args.question, args.top_k
    )
    print(f"Question: {report['question']}")
    print(
        "Dense/Lexical overlap: "
        f"{report['overlap']['dense_lexical_shared_count']}/{report['overlap']['dense_lexical_union_count']}"
    )
    for backend in ("dense", "lexical", "hybrid"):
        print(f"\n{backend}:")
        for item in report[backend]:
            print(f"- [{item['rank']}] {item['source']} | {item['context']} | score={item['score']}")
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report saved -> {destination}")


def evaluate(index: TfidfIndex, cases: list[dict], top_k: int) -> dict:
    if not cases:
        raise ValueError("Evaluation cases must not be empty")
    hits = 0
    records = []
    started = time.perf_counter()
    for case in cases:
        case_started = time.perf_counter()
        results = index.search(case["question"], top_k)
        sources = [result.source for result in results]
        expected_sources = case.get("expected_sources") or [case["expected_source"]]
        hit = any(source in sources for source in expected_sources)
        if hit:
            hits += 1
        records.append(
            {
                "id": case.get("id"),
                "category": case.get("category", "uncategorized"),
                "question": case["question"],
                "expected_sources": expected_sources,
                "retrieved_sources": sources,
                "hit": hit,
                "latency_ms": round((time.perf_counter() - case_started) * 1000, 3),
            }
        )
    total = len(cases)
    category_metrics: dict[str, dict[str, int | float]] = {}
    for record in records:
        category = record["category"]
        summary = category_metrics.setdefault(category, {"hits": 0, "total": 0})
        summary["total"] += 1
        summary["hits"] += int(record["hit"])
    for summary in category_metrics.values():
        summary["score"] = round(summary["hits"] / summary["total"], 4)
    return {
        "metric": f"recall@{top_k}",
        "score": hits / total,
        "hits": hits,
        "total": total,
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "category_metrics": category_metrics,
        "cases": records,
    }


def benchmark_retrieval(index: TfidfIndex, cases: list[dict], top_k: int, repeats: int) -> dict:
    if not cases:
        raise ValueError("Benchmark cases must not be empty")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    # Load models and allocate first-use state before measuring steady-state latency.
    index.search(cases[0]["question"], top_k)
    latencies = []
    for _ in range(repeats):
        for case in cases:
            started = time.perf_counter()
            index.search(case["question"], top_k)
            latencies.append((time.perf_counter() - started) * 1000)
    ordered = sorted(latencies)
    p95_index = max(0, round((len(ordered) - 1) * 0.95))
    return {
        "metric": "steady_state_retrieval_latency_ms",
        "queries": len(latencies),
        "top_k": top_k,
        "repeats": repeats,
        "latency_ms": {
            "median": round(median(latencies), 3),
            "p95": round(ordered[p95_index], 3),
            "max": round(max(latencies), 3),
        },
    }


def print_category_metrics(report: dict) -> None:
    metrics = report.get("category_metrics", {})
    if len(metrics) <= 1 and "uncategorized" in metrics:
        return
    print("By category:")
    for category, summary in sorted(metrics.items()):
        print(f"- {category}: {summary['score']:.1%} ({summary['hits']}/{summary['total']})")


def command_eval(args: argparse.Namespace) -> None:
    index = load_index(Path(args.index))
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    report = evaluate(index, cases, args.top_k)
    print(f"recall@{args.top_k}={report['score']:.1%} ({report['hits']}/{report['total']})")
    print_category_metrics(report)
    failures = [record for record in report["cases"] if not record["hit"]]
    if failures:
        print("Failures:")
        for record in failures:
            print(f"- {record['question']} -> expected one of {', '.join(record['expected_sources'])}")
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report saved -> {destination}")


def command_embedding_eval(args: argparse.Namespace) -> None:
    index = load_embedding_index(Path(args.index))
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    report = evaluate(index, cases, args.top_k)
    print(f"embedding recall@{args.top_k}={report['score']:.1%} ({report['hits']}/{report['total']})")
    print_category_metrics(report)
    failures = [record for record in report["cases"] if not record["hit"]]
    if failures:
        print("Failures:")
        for record in failures:
            print(f"- {record['question']} -> expected one of {', '.join(record['expected_sources'])}")
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report saved -> {destination}")


def command_hybrid_eval(args: argparse.Namespace) -> None:
    index = load_hybrid_retriever(args.index, args.lexical_index)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    report = evaluate(index, cases, args.top_k)
    print(f"hybrid recall@{args.top_k}={report['score']:.1%} ({report['hits']}/{report['total']})")
    print_category_metrics(report)
    failures = [record for record in report["cases"] if not record["hit"]]
    if failures:
        print("Failures:")
        for record in failures:
            print(f"- {record['question']} -> expected one of {', '.join(record['expected_sources'])}")
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report saved -> {destination}")


def command_benchmark(args: argparse.Namespace) -> None:
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    report = benchmark_retrieval(
        load_hybrid_retriever(args.index, args.lexical_index), cases, args.top_k, args.repeats
    )
    latency = report["latency_ms"]
    print(
        f"steady-state latency: median={latency['median']} ms "
        f"p95={latency['p95']} ms max={latency['max']} ms ({report['queries']} queries)"
    )
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report saved -> {destination}")


def command_rerank_query(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    retriever = RerankingRetriever(
        load_embedding_index(Path(args.index)),
        model_name=args.model,
        candidate_k=args.candidate_k,
        local_files_only=not args.allow_download,
    )
    results = retriever.search(args.question, args.top_k)
    print_results(results, (time.perf_counter() - started) * 1000)


def command_rerank_eval(args: argparse.Namespace) -> None:
    retriever = RerankingRetriever(
        load_embedding_index(Path(args.index)),
        model_name=args.model,
        candidate_k=args.candidate_k,
        local_files_only=not args.allow_download,
    )
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    report = evaluate(retriever, cases, args.top_k)
    print(f"reranked recall@{args.top_k}={report['score']:.1%} ({report['hits']}/{report['total']})")
    print_category_metrics(report)
    failures = [record for record in report["cases"] if not record["hit"]]
    if failures:
        print("Failures:")
        for record in failures:
            print(f"- {record['question']} -> expected one of {', '.join(record['expected_sources'])}")
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report saved -> {destination}")


def load_hybrid_reranker(args: argparse.Namespace) -> RerankingRetriever:
    """Build the production retrieval chain: hybrid recall followed by reranking."""
    return RerankingRetriever(
        load_hybrid_retriever(args.index, args.lexical_index),
        model_name=args.model,
        candidate_k=args.candidate_k,
        local_files_only=not args.allow_download,
    )


def command_hybrid_rerank_query(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    results = load_hybrid_reranker(args).search(args.question, args.top_k)
    print_results(results, (time.perf_counter() - started) * 1000)


def command_hybrid_rerank_eval(args: argparse.Namespace) -> None:
    retriever = load_hybrid_reranker(args)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    report = evaluate(retriever, cases, args.top_k)
    print(f"hybrid + reranker recall@{args.top_k}={report['score']:.1%} ({report['hits']}/{report['total']})")
    print_category_metrics(report)
    failures = [record for record in report["cases"] if not record["hit"]]
    if failures:
        print("Failures:")
        for record in failures:
            print(f"- {record['question']} -> expected one of {', '.join(record['expected_sources'])}")
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report saved -> {destination}")


def evaluate_abstention(index: TfidfIndex, cases: list[dict], top_k: int) -> dict:
    if not cases:
        raise ValueError("Abstention evaluation cases must not be empty")
    records = []
    for case in cases:
        evidence = index.search(case["question"], top_k)
        assessment = assess_evidence(evidence, case["question"])
        expected_refusal = bool(case["expect_refusal"])
        predicted_refusal = not assessment.sufficient
        records.append(
            {
                "id": case.get("id"),
                "question": case["question"],
                "expected_refusal": expected_refusal,
                "predicted_refusal": predicted_refusal,
                "correct": expected_refusal == predicted_refusal,
                "assessment": assessment.to_dict(),
                "retrieved_sources": [item.source for item in evidence],
            }
        )
    true_positive = sum(item["expected_refusal"] and item["predicted_refusal"] for item in records)
    false_positive = sum(not item["expected_refusal"] and item["predicted_refusal"] for item in records)
    false_negative = sum(item["expected_refusal"] and not item["predicted_refusal"] for item in records)
    total = len(records)
    return {
        "metric": "abstention_accuracy",
        "score": sum(item["correct"] for item in records) / total,
        "correct": sum(item["correct"] for item in records),
        "total": total,
        "refusal_precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0,
        "refusal_recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0,
        "cases": records,
    }


def command_abstention_eval(args: argparse.Namespace) -> None:
    retriever = load_hybrid_retriever(args.index, args.lexical_index)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    report = evaluate_abstention(retriever, cases, args.top_k)
    print(f"abstention accuracy={report['score']:.1%} ({report['correct']}/{report['total']})")
    print(f"refusal precision={report['refusal_precision']:.1%} recall={report['refusal_recall']:.1%}")
    failures = [record for record in report["cases"] if not record["correct"]]
    if failures:
        print("Failures:")
        for record in failures:
            print(f"- {record['question']} -> predicted_refusal={record['predicted_refusal']}")
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report saved -> {destination}")


def command_answer(args: argparse.Namespace) -> None:
    evidence = load_hybrid_retriever(args.index, args.lexical_index).search(args.question, args.top_k)
    answer = DeepSeekGenerator.from_environment(model=args.model).answer(args.question, evidence)
    print(f"\n回答：\n{answer}\n\n检索证据：")
    for item in evidence:
        print(f"- {item.source} | {item.context} | score={item.score}")


def command_answer_eval(args: argparse.Namespace) -> None:
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    index, generator, records = load_hybrid_retriever(args.index, args.lexical_index), DeepSeekGenerator.from_environment(args.model), []
    for case in cases:
        evidence = index.search(case["question"], args.top_k)
        answer = generator.answer(case["question"], evidence)
        check = check_answer(answer, case["expected_terms"], case["expected_sources"])
        records.append({"question": case["question"], "answer": answer, **check})
    passed = sum(item["terms_pass"] and item["citation_pass"] for item in records)
    report = {"score": passed / len(records), "passed": passed, "total": len(records), "cases": records}
    print(f"answer quality={report['score']:.1%} ({passed}/{len(records)})")
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def command_feedback_report(args: argparse.Namespace) -> None:
    report = build_feedback_report(
        AnswerHistory(Path(args.history)).all(),
        FeedbackHistory(Path(args.feedback)).all(),
    )
    queue = report["review_queue"][: args.limit]
    print(f"反馈总数：{report['total_feedback']}")
    print(f"有帮助：{report['ratings']['up']}；需要改进：{report['ratings']['down']}")
    if report["reasons"]:
        print("点踩原因：")
        for reason, count in sorted(report["reasons"].items(), key=lambda item: (-item[1], item[0])):
            print(f"- {REASON_LABELS.get(reason, reason)}：{count}")
    if queue:
        print("待复盘问题：")
        for number, item in enumerate(queue, 1):
            print(f"[{number}] {item['reason_label']} | {item['question']}")
            if item["sources"]:
                print(f"    证据：{', '.join(item['sources'])}")
            if item["note"]:
                print(f"    备注：{item['note']}")
    else:
        print("暂无点踩记录。")
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已保存：{destination}")


def command_trace_report(args: argparse.Namespace) -> None:
    report = build_trace_report(AnswerHistory(Path(args.history)).all(), args.limit)
    print(f"Trace 总数：{report['total_traces']}")
    print(f"平均耗时：{report['latency_ms']['average']} ms；P95：{report['latency_ms']['p95']} ms")
    print(f"缺少完整追踪字段的旧记录：{report['incomplete_trace_count']}")
    if report["slow_traces"]:
        print("最慢请求：")
        for item in report["slow_traces"]:
            print(f"- {item['latency_ms']} ms | {item['trace_id']} | {item['question']}")
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已保存：{destination}")


def command_grounding_report(args: argparse.Namespace) -> None:
    report = build_grounding_report(AnswerHistory(Path(args.history)).all(), args.limit)
    print(f"Grounding traces: {report['evaluated_traces']}/{report['total_traces']}")
    print(f"Refused: {report['refused_count']} | citation valid rate: {report['citation_valid_rate']:.1%}")
    if report["invalid_citation_examples"]:
        print("Invalid citation examples:")
        for item in report["invalid_citation_examples"]:
            print(f"- {item['trace_id']} | {item['question']} | unsupported={item['unsupported_sources']}")
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report saved -> {destination}")


def command_judge_eval(args: argparse.Namespace) -> None:
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if not cases:
        raise ValueError("评测集不能为空")
    retriever = load_hybrid_retriever(args.index, args.lexical_index)
    generator = DeepSeekGenerator.from_environment(args.model)
    judge = GroundedAnswerJudge.from_environment(args.judge_model)
    records = []
    for case in cases:
        evidence = retriever.search(case["question"], args.top_k)
        generated = generator.answer_with_usage(case["question"], evidence)
        judgement = judge.evaluate(case["question"], generated.content, evidence)
        deterministic = check_answer(
            generated.content,
            case.get("expected_terms", []),
            case.get("expected_sources", []),
        )
        records.append(
            {
                "question": case["question"],
                "answer": generated.content,
                "evidence": [item.__dict__ for item in evidence],
                "generation_usage": generated.usage,
                "deterministic_checks": deterministic,
                "judge": judgement,
            }
        )
    average = sum(item["judge"]["weighted_score"] for item in records) / len(records)
    report = {
        "judge_prompt_version": "grounded-rubric-v1",
        "generator_model": args.model,
        "judge_model": args.judge_model,
        "average_weighted_score": round(average, 3),
        "pass_threshold": args.pass_threshold,
        "passed": sum(item["judge"]["weighted_score"] >= args.pass_threshold for item in records),
        "total": len(records),
        "cases": records,
    }
    print(
        f"judge score={report['average_weighted_score']:.3f}/5 "
        f"pass@{args.pass_threshold:g}={report['passed']}/{report['total']}"
    )
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Judge report saved -> {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Campus RAG offline retrieval baseline")
    commands = parser.add_subparsers(required=True)
    index = commands.add_parser("index")
    index.add_argument("--docs", required=True)
    index.add_argument("--output", required=True)
    index.set_defaults(handler=command_index)
    docs_lint = commands.add_parser("docs-lint", help="Check local source files before indexing")
    docs_lint.add_argument("--docs", required=True)
    docs_lint.add_argument("--max-chunk-chars", type=int, default=2400)
    docs_lint.add_argument("--report", help="Optional JSON path for the lint report")
    docs_lint.add_argument("--fail-on-warning", action="store_true")
    docs_lint.set_defaults(handler=command_docs_lint)
    index_status = commands.add_parser("index-status", help="Check whether local indexes match current source documents")
    index_status.add_argument("--docs", default="data/docs")
    index_status.add_argument("--index", default="data/embedding_index.json")
    index_status.add_argument("--lexical-index", default="data/index.json")
    index_status.add_argument("--report", help="Optional JSON path for index status")
    index_status.add_argument("--require-fresh", action="store_true")
    index_status.set_defaults(handler=command_index_status)
    quality_gate = commands.add_parser("quality-gate", help="Apply release thresholds to retrieval and abstention reports")
    quality_gate.add_argument("--docs", default="data/docs")
    quality_gate.add_argument("--index", default="data/embedding_index.json")
    quality_gate.add_argument("--lexical-index", default="data/index.json")
    quality_gate.add_argument("--retrieval-report", required=True)
    quality_gate.add_argument("--abstention-report", required=True)
    quality_gate.add_argument("--report", required=True)
    quality_gate.add_argument("--min-recall", type=float, default=1.0)
    quality_gate.add_argument("--min-abstention-accuracy", type=float, default=1.0)
    quality_gate.set_defaults(handler=command_quality_gate)
    query = commands.add_parser("query")
    query.add_argument("--index", required=True)
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=3)
    query.set_defaults(handler=command_query)

    embedding_index = commands.add_parser("embedding-index", help="Build a dense vector index on CPU")
    embedding_index.add_argument("--docs", required=True)
    embedding_index.add_argument("--output", required=True)
    embedding_index.add_argument("--model", default=DEFAULT_MODEL)
    embedding_index.set_defaults(handler=command_embedding_index)
    embedding_query = commands.add_parser("embedding-query", help="Query a dense vector index")
    embedding_query.add_argument("--index", required=True)
    embedding_query.add_argument("--question", required=True)
    embedding_query.add_argument("--top-k", type=int, default=3)
    embedding_query.set_defaults(handler=command_embedding_query)
    hybrid_query = commands.add_parser("hybrid-query", help="Query with dense and TF-IDF retrieval together")
    hybrid_query.add_argument("--index", required=True, help="Path to an embedding index")
    hybrid_query.add_argument("--lexical-index", default="data/index.json")
    hybrid_query.add_argument("--question", required=True)
    hybrid_query.add_argument("--top-k", type=int, default=3)
    hybrid_query.set_defaults(handler=command_hybrid_query)
    diagnose = commands.add_parser("diagnose-query", help="Compare dense, lexical, and hybrid retrieval for one question")
    diagnose.add_argument("--index", required=True, help="Path to an embedding index")
    diagnose.add_argument("--lexical-index", default="data/index.json")
    diagnose.add_argument("--question", required=True)
    diagnose.add_argument("--top-k", type=int, default=3)
    diagnose.add_argument("--report", help="Optional JSON path for the diagnostic")
    diagnose.set_defaults(handler=command_diagnose_query)
    evaluate = commands.add_parser("eval")
    evaluate.add_argument("--index", required=True)
    evaluate.add_argument("--cases", required=True)
    evaluate.add_argument("--top-k", type=int, default=3)
    evaluate.add_argument("--report", help="Optional JSON path for per-case evaluation results")
    evaluate.set_defaults(handler=command_eval)
    embedding_evaluate = commands.add_parser("embedding-eval", help="Evaluate a dense vector index")
    embedding_evaluate.add_argument("--index", required=True)
    embedding_evaluate.add_argument("--cases", required=True)
    embedding_evaluate.add_argument("--top-k", type=int, default=3)
    embedding_evaluate.add_argument("--report", help="Optional JSON path for per-case evaluation results")
    embedding_evaluate.set_defaults(handler=command_embedding_eval)
    hybrid_evaluate = commands.add_parser("hybrid-eval", help="Evaluate dense and TF-IDF retrieval together")
    hybrid_evaluate.add_argument("--index", required=True, help="Path to an embedding index")
    hybrid_evaluate.add_argument("--lexical-index", default="data/index.json")
    hybrid_evaluate.add_argument("--cases", required=True)
    hybrid_evaluate.add_argument("--top-k", type=int, default=3)
    hybrid_evaluate.add_argument("--report", help="Optional JSON path for per-case evaluation results")
    hybrid_evaluate.set_defaults(handler=command_hybrid_eval)
    benchmark = commands.add_parser("benchmark", help="Measure warm retrieval latency without calling an LLM")
    benchmark.add_argument("--index", required=True, help="Path to an embedding index")
    benchmark.add_argument("--lexical-index", default="data/index.json")
    benchmark.add_argument("--cases", required=True)
    benchmark.add_argument("--report", required=True)
    benchmark.add_argument("--top-k", type=int, default=3)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.set_defaults(handler=command_benchmark)
    rerank_query = commands.add_parser("rerank-query", help="Rerank dense-retrieval candidates on CPU")
    rerank_query.add_argument("--index", required=True, help="Path to an embedding index")
    rerank_query.add_argument("--question", required=True)
    rerank_query.add_argument("--top-k", type=int, default=3)
    rerank_query.add_argument("--candidate-k", type=int, default=10)
    rerank_query.add_argument("--model", default=DEFAULT_RERANKER)
    rerank_query.add_argument("--allow-download", action="store_true", help="Allow the first model download")
    rerank_query.set_defaults(handler=command_rerank_query)
    rerank_evaluate = commands.add_parser("rerank-eval", help="Evaluate dense retrieval followed by reranking")
    rerank_evaluate.add_argument("--index", required=True, help="Path to an embedding index")
    rerank_evaluate.add_argument("--cases", required=True)
    rerank_evaluate.add_argument("--top-k", type=int, default=3)
    rerank_evaluate.add_argument("--candidate-k", type=int, default=10)
    rerank_evaluate.add_argument("--model", default=DEFAULT_RERANKER)
    rerank_evaluate.add_argument("--allow-download", action="store_true", help="Allow the first model download")
    rerank_evaluate.add_argument("--report", help="Optional JSON path for per-case evaluation results")
    rerank_evaluate.set_defaults(handler=command_rerank_eval)
    hybrid_rerank_query = commands.add_parser("hybrid-rerank-query", help="Rerank hybrid-retrieval candidates on CPU")
    hybrid_rerank_query.add_argument("--index", required=True, help="Path to an embedding index")
    hybrid_rerank_query.add_argument("--lexical-index", default="data/index.json")
    hybrid_rerank_query.add_argument("--question", required=True)
    hybrid_rerank_query.add_argument("--top-k", type=int, default=3)
    hybrid_rerank_query.add_argument("--candidate-k", type=int, default=10)
    hybrid_rerank_query.add_argument("--model", default=DEFAULT_RERANKER)
    hybrid_rerank_query.add_argument("--allow-download", action="store_true", help="Allow the first model download")
    hybrid_rerank_query.set_defaults(handler=command_hybrid_rerank_query)
    hybrid_rerank_evaluate = commands.add_parser("hybrid-rerank-eval", help="Evaluate hybrid retrieval followed by reranking")
    hybrid_rerank_evaluate.add_argument("--index", required=True, help="Path to an embedding index")
    hybrid_rerank_evaluate.add_argument("--lexical-index", default="data/index.json")
    hybrid_rerank_evaluate.add_argument("--cases", required=True)
    hybrid_rerank_evaluate.add_argument("--top-k", type=int, default=3)
    hybrid_rerank_evaluate.add_argument("--candidate-k", type=int, default=10)
    hybrid_rerank_evaluate.add_argument("--model", default=DEFAULT_RERANKER)
    hybrid_rerank_evaluate.add_argument("--allow-download", action="store_true", help="Allow the first model download")
    hybrid_rerank_evaluate.add_argument("--report", help="Optional JSON path for per-case evaluation results")
    hybrid_rerank_evaluate.set_defaults(handler=command_hybrid_rerank_eval)
    abstention_evaluate = commands.add_parser("abstention-eval", help="Evaluate evidence-gated refusal without calling an LLM")
    abstention_evaluate.add_argument("--index", required=True, help="Path to an embedding index")
    abstention_evaluate.add_argument("--lexical-index", default="data/index.json")
    abstention_evaluate.add_argument("--cases", required=True)
    abstention_evaluate.add_argument("--report", required=True)
    abstention_evaluate.add_argument("--top-k", type=int, default=3)
    abstention_evaluate.set_defaults(handler=command_abstention_eval)
    answer = commands.add_parser("answer", help="Answer from embedding evidence through DeepSeek")
    answer.add_argument("--index", required=True, help="Path to an embedding index")
    answer.add_argument("--question", required=True)
    answer.add_argument("--top-k", type=int, default=3)
    answer.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL)
    answer.add_argument("--lexical-index", default="data/index.json")
    answer.set_defaults(handler=command_answer)
    answer_eval = commands.add_parser("answer-eval", help="Evaluate grounded DeepSeek answers")
    answer_eval.add_argument("--index", required=True); answer_eval.add_argument("--cases", required=True)
    answer_eval.add_argument("--report", required=True); answer_eval.add_argument("--top-k", type=int, default=3)
    answer_eval.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL); answer_eval.set_defaults(handler=command_answer_eval)
    answer_eval.add_argument("--lexical-index", default="data/index.json")
    feedback_report = commands.add_parser("feedback-report", help="Summarize local user feedback without calling an LLM")
    feedback_report.add_argument("--history", default="logs/answer_history.jsonl")
    feedback_report.add_argument("--feedback", default="logs/feedback.jsonl")
    feedback_report.add_argument("--limit", type=int, default=20)
    feedback_report.add_argument("--report", help="Optional JSON path for the feedback report")
    feedback_report.set_defaults(handler=command_feedback_report)
    trace_report = commands.add_parser("trace-report", help="Summarize local answer traces without calling an LLM")
    trace_report.add_argument("--history", default="logs/answer_history.jsonl")
    trace_report.add_argument("--limit", type=int, default=10)
    trace_report.add_argument("--report", help="Optional JSON path for the trace report")
    trace_report.set_defaults(handler=command_trace_report)
    grounding_report = commands.add_parser("grounding-report", help="Summarize citation validation from local answer traces")
    grounding_report.add_argument("--history", default="logs/answer_history.jsonl")
    grounding_report.add_argument("--limit", type=int, default=20)
    grounding_report.add_argument("--report", help="Optional JSON path for the grounding report")
    grounding_report.set_defaults(handler=command_grounding_report)
    judge_eval = commands.add_parser("judge-eval", help="Score generated RAG answers against retrieved evidence using an LLM judge")
    judge_eval.add_argument("--index", required=True, help="Path to an embedding index")
    judge_eval.add_argument("--lexical-index", default="data/index.json")
    judge_eval.add_argument("--cases", required=True)
    judge_eval.add_argument("--report", required=True)
    judge_eval.add_argument("--top-k", type=int, default=3)
    judge_eval.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL, help="Generator model")
    judge_eval.add_argument("--judge-model", default=DEFAULT_DEEPSEEK_MODEL, help="Judge model; prefer a different model when available")
    judge_eval.add_argument("--pass-threshold", type=float, default=4.0)
    judge_eval.set_defaults(handler=command_judge_eval)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.handler(parsed)

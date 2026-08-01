from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .embeddings import DEFAULT_MODEL, EmbeddingIndex
from .feedback_report import REASON_LABELS, build_feedback_report
from .generation import DEFAULT_MODEL as DEFAULT_DEEPSEEK_MODEL, DeepSeekGenerator, check_answer
from .history import AnswerHistory, FeedbackHistory
from .hybrid import HybridRetriever
from .judge import GroundedAnswerJudge
from .reranking import DEFAULT_RERANKER, RerankingRetriever
from .retrieval import TfidfIndex, load_chunks
from .trace_report import build_trace_report


def load_index(path: Path) -> TfidfIndex:
    return TfidfIndex.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_embedding_index(path: Path) -> EmbeddingIndex:
    return EmbeddingIndex.from_dict(json.loads(path.read_text(encoding="utf-8")))


def command_index(args: argparse.Namespace) -> None:
    index = TfidfIndex.build(load_chunks(Path(args.docs)))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed {len(index.chunks)} chunks -> {destination}")


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
    index = EmbeddingIndex.build(load_chunks(Path(args.docs)), model_name=args.model)
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
                "question": case["question"],
                "expected_sources": expected_sources,
                "retrieved_sources": sources,
                "hit": hit,
                "latency_ms": round((time.perf_counter() - case_started) * 1000, 3),
            }
        )
    total = len(cases)
    return {
        "metric": f"recall@{top_k}",
        "score": hits / total,
        "hits": hits,
        "total": total,
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "cases": records,
    }


def command_eval(args: argparse.Namespace) -> None:
    index = load_index(Path(args.index))
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    report = evaluate(index, cases, args.top_k)
    print(f"recall@{args.top_k}={report['score']:.1%} ({report['hits']}/{report['total']})")
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

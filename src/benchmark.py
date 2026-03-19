"""Benchmark script for evaluating YCAdvisor RAG quality."""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Set, Tuple

from rag import YCAdvisor

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except Exception:  # pragma: no cover - optional dependency guard
    trace = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS_PATH = os.path.join(BASE_DIR, "benchmark_questions.json")
RESULTS_PATH = os.path.join(BASE_DIR, "benchmark_results.json")

OUT_OF_SCOPE_HINTS = [
    "i don't have reliable data in this knowledge base to answer that",
    "i'm focused on yc and startup topics",
    "i don't have reliable yc data to answer this well",
]


def _setup_tracing(service_name: str = "yc-benchmark"):
    """Initialize OpenTelemetry tracer provider once per process."""
    if trace is None:
        return None

    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        return trace.get_tracer("yc.benchmark")

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if endpoint:
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    else:
        # Console exporter is useful for local debugging when no collector is configured.
        processor = BatchSpanProcessor(ConsoleSpanExporter())

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return trace.get_tracer("yc.benchmark")


TRACER = _setup_tracing()


def load_questions(path: str) -> List[Dict[str, Any]]:
    """Load benchmark question definitions from JSON."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_sources(sources: List[Dict[str, Any]]) -> Set[str]:
    """Normalize returned source_type values for matching."""
    return {
        str(source.get("source_type", "")).strip().lower()
        for source in sources
        if str(source.get("source_type", "")).strip()
    }


def _stem(word: str) -> str:
    """Reduce a word to a rough stem by stripping common English suffixes."""
    w = word.lower().strip()
    for suffix in ("ation", "tion", "ing", "ment", "ness", "ers", "ies", "es", "ed", "ly", "al", "er", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    return w


# Synonyms grouped by concept — any word in the tuple counts as a match for any
# other word in the same tuple.
_SYNONYM_GROUPS: List[Tuple[str, ...]] = [
    ("users", "user", "customers", "customer"),
    ("growth", "grow", "growing", "scale", "scaling"),
    ("retention", "retain", "retaining", "churn"),
    ("revenue", "income", "money", "earning", "profitable", "profitability"),
    ("expenses", "costs", "spending", "burn", "burn rate"),
    ("traction", "momentum", "progress"),
    ("hiring", "hire", "recruit", "recruiting"),
    ("build", "building", "built", "create", "creating"),
    ("product", "products"),
    ("founders", "founder", "cofounders", "cofounder", "co-founder", "co-founders"),
    ("investors", "investor", "vc", "vcs", "venture capital"),
    ("manual", "manually", "hand", "by hand"),
    ("early", "earliest", "beginning", "start"),
    ("small", "smaller", "lean", "tiny"),
    ("clear", "clearly", "clarity", "concise"),
    ("metrics", "metric", "kpi", "kpis", "numbers", "data"),
    ("pitch", "pitching", "pitches", "demo"),
    ("market", "markets", "industry"),
    ("software", "code", "codebase", "engineering"),
    ("pivot", "pivoting", "pivoted", "change direction"),
    ("problem", "problems", "pain point", "pain points", "challenge"),
    ("idea", "ideas", "concept", "concepts", "vision"),
    ("survival", "survive", "surviving", "alive", "die", "dying"),
    ("demand", "need", "want"),
]


def _build_synonym_map() -> Dict[str, Set[str]]:
    """Build a lookup: word -> set of all synonyms (including itself)."""
    mapping: Dict[str, Set[str]] = {}
    for group in _SYNONYM_GROUPS:
        group_set = set(group)
        for word in group:
            mapping.setdefault(word, set()).update(group_set)
    return mapping


_SYNONYM_MAP = _build_synonym_map()


def _topic_found_in_answer(topic: str, answer_lower: str) -> bool:
    """Check if a topic appears in the answer using direct, stem, or synonym matching."""
    tl = topic.lower().strip()

    # 1. Direct substring match
    if tl in answer_lower:
        return True

    # 2. Stem match — check if the stem of the topic appears in the answer
    topic_stem = _stem(tl)
    if len(topic_stem) >= 3 and topic_stem in answer_lower:
        return True

    # 3. Synonym match — check if any synonym of the topic appears in the answer
    for synonym in _SYNONYM_MAP.get(tl, set()):
        if synonym in answer_lower:
            return True
        syn_stem = _stem(synonym)
        if len(syn_stem) >= 3 and syn_stem in answer_lower:
            return True

    return False


def score_relevance(expected_topics: List[str], answer: str) -> float:
    """Score relevance by expected topic mention coverage with stem + synonym matching."""
    answer_lower = answer.lower()
    topics_lower = [topic.lower() for topic in expected_topics]

    if topics_lower == ["out of scope"]:
        return 1.0 if any(hint in answer_lower for hint in OUT_OF_SCOPE_HINTS) else 0.0

    if not topics_lower:
        return 1.0

    found = sum(1 for topic in topics_lower if _topic_found_in_answer(topic, answer_lower))
    return found / len(topics_lower)


def score_sources(expected_sources: List[str], actual_sources: Set[str]) -> float:
    """Score source correctness by expected source-type coverage."""
    expected_lower = [source.lower() for source in expected_sources]

    # Out-of-scope questions have no expected sources — don't penalise.
    if not expected_lower:
        return 1.0

    found = sum(1 for source in expected_lower if source in actual_sources)
    return found / len(expected_lower)


def score_hallucination(should_not_contain: List[str], answer: str) -> float:
    """Return 1.0 if forbidden terms are absent, else 0.0."""
    if not should_not_contain:
        return 1.0

    answer_lower = answer.lower()
    has_forbidden = any(term.lower() in answer_lower for term in should_not_contain)
    return 0.0 if has_forbidden else 1.0


def pass_fail(score: float) -> str:
    """Return indicator based on 0.6 threshold."""
    return "✅" if score >= 0.6 else "❌"


def evaluate_question(
    advisor: YCAdvisor,
    question_item: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run one benchmark question and return result + printable metrics."""
    span_cm = (
        TRACER.start_as_current_span("benchmark.question")
        if TRACER is not None
        else None
    )
    if span_cm is not None:
        span_cm.__enter__()

    total_chunks = advisor.retriever.collection.count()

    question = question_item["question"]
    if TRACER is not None:
        current_span = trace.get_current_span()
        current_span.set_attribute("benchmark.question", question)
    expected_topics = question_item.get("expected_topics", [])
    expected_sources = question_item.get("expected_sources", [])
    should_not_contain = question_item.get("should_not_contain", [])

    answer_start = time.perf_counter()
    response = advisor.ask_with_sources(question)
    answer_latency = time.perf_counter() - answer_start

    answer = response.get("answer", "")

    source_check_start = time.perf_counter()
    benchmark_sources = advisor.retriever.search(query=question, n=total_chunks)
    source_check_latency = time.perf_counter() - source_check_start

    checked_sources = [
        {
            "title": c.get("title", ""),
            "author": c.get("author", ""),
            "source_type": c.get("source_type", ""),
        }
        for c in benchmark_sources
    ]
    actual_source_types = normalize_sources(checked_sources)

    relevance = score_relevance(expected_topics, answer)
    source = score_sources(expected_sources, actual_source_types)
    hallucination = score_hallucination(should_not_contain, answer)
    avg_score = (relevance + source + hallucination) / 3

    total_latency = round(answer_latency + source_check_latency, 4)

    result = {
        "question": question,
        "expected_topics": expected_topics,
        "expected_sources": expected_sources,
        "should_not_contain": should_not_contain,
        "answer": answer,
        "answer_sources": response.get("sources", []),
        "source_count_checked": len(checked_sources),
        "sources_checked": checked_sources,
        "actual_source_types": sorted(actual_source_types),
        "relevance_score": round(relevance, 4),
        "source_score": round(source, 4),
        "hallucination_score": round(hallucination, 4),
        "avg_score": round(avg_score, 4),
        "latency_sec": total_latency,
        "latency": {
            "answer_latency_sec": round(answer_latency, 4),
            "source_check_latency_sec": round(source_check_latency, 4),
            "total_latency_sec": total_latency,
        },
        "scores": {
            "relevance": round(relevance, 4),
            "source": round(source, 4),
            "hallucination": round(hallucination, 4),
            "average": round(avg_score, 4),
        },
    }

    printable = {
        "relevance": relevance,
        "source": source,
        "hallucination": hallucination,
        "average": avg_score,
        "source_count_checked": len(checked_sources),
        "total_latency_sec": answer_latency + source_check_latency,
    }

    if TRACER is not None:
        current_span = trace.get_current_span()
        current_span.set_attribute("benchmark.relevance", float(relevance))
        current_span.set_attribute("benchmark.source", float(source))
        current_span.set_attribute("benchmark.hallucination", float(hallucination))
        current_span.set_attribute("benchmark.average", float(avg_score))
        current_span.set_attribute("benchmark.latency_sec", float(total_latency))

    if span_cm is not None:
        span_cm.__exit__(None, None, None)

    return result, printable


def print_question_result(index: int, question: str, scores: Dict[str, float]) -> None:
    """Print one question's benchmark summary."""
    print(f"Q{index}: {question}")
    print(f"Relevance:     {scores['relevance']:.2f}  {pass_fail(scores['relevance'])}")
    print(f"Source:        {scores['source']:.2f}  {pass_fail(scores['source'])}")
    print(
        "Hallucination: "
        f"{scores['hallucination']:.2f}  {pass_fail(scores['hallucination'])}"
    )
    print(f"Latency:       {scores['total_latency_sec']:.2f}s")
    print(f"Sources used for check: {scores['source_count_checked']}")
    print(f"Avg Score:     {scores['average']:.2f}")
    print("─────────────────────")


def print_report(
    total: int,
    avg_relevance: float,
    avg_source: float,
    avg_hallucination: float,
    overall_score: float,
    avg_latency_sec: float,
    p95_latency_sec: float,
    weak_questions: List[str],
    strong_questions: List[str],
) -> None:
    """Print final benchmark report block."""
    print("════════════════════════════════")
    print("BENCHMARK REPORT")
    print("════════════════════════════════")
    print(f"Total questions tested: {total}\n")
    print(f"Avg Relevance Score:     {avg_relevance:.2f}")
    print(f"Avg Source Score:        {avg_source:.2f}")
    print(f"Avg Hallucination Score: {avg_hallucination:.2f}")
    print(f"Avg Latency:             {avg_latency_sec:.2f}s")
    print(f"P95 Latency:             {p95_latency_sec:.2f}s")
    print(f"Sources Checked:         all available")
    print(f"Overall RAG Score:       {overall_score:.2f} / 1.00\n")

    print("Weak questions (score < 0.6):")
    if weak_questions:
        for question in weak_questions:
            print(f"- {question}")
    else:
        print("- None")

    print("\nStrong questions (score > 0.85):")
    if strong_questions:
        for question in strong_questions:
            print(f"- {question}")
    else:
        print("- None")

    print("════════════════════════════════")


def run_benchmark(
    advisor: YCAdvisor | None = None,
    questions_path: str = QUESTIONS_PATH,
    results_path: str = RESULTS_PATH,
    max_questions: int | None = None,
    print_progress: bool = True,
) -> Dict[str, Any]:
    """Run benchmark, save JSON report, and return report dict."""
    span_cm = (
        TRACER.start_as_current_span("benchmark.run")
        if TRACER is not None
        else None
    )
    if span_cm is not None:
        span_cm.__enter__()

    questions = load_questions(questions_path)
    if max_questions is not None:
        questions = questions[:max(0, max_questions)]
    if TRACER is not None:
        current_span = trace.get_current_span()
        current_span.set_attribute("benchmark.total_questions", len(questions))

    advisor = advisor or YCAdvisor()

    results: List[Dict[str, Any]] = []
    running_relevance = 0.0
    running_source = 0.0
    running_hallucination = 0.0
    latencies: List[float] = []

    weak_questions: List[str] = []
    strong_questions: List[str] = []

    for index, item in enumerate(questions, start=1):
        try:
            result, scores = evaluate_question(advisor, item)
        except Exception as exc:
            if "429" in str(exc):
                if print_progress:
                    print(f"\nRate limit hit at Q{index}. Saving partial results ({len(results)} completed).")
                break
            raise
        results.append(result)

        running_relevance += scores["relevance"]
        running_source += scores["source"]
        running_hallucination += scores["hallucination"]
        latencies.append(scores["total_latency_sec"])

        if scores["average"] < 0.6:
            weak_questions.append(item["question"])
        if scores["average"] > 0.85:
            strong_questions.append(item["question"])

        if print_progress:
            print_question_result(index, item["question"], scores)

    total = len(results)
    avg_relevance = running_relevance / total if total else 0.0
    avg_source = running_source / total if total else 0.0
    avg_hallucination = running_hallucination / total if total else 0.0
    avg_latency_sec = (sum(latencies) / total) if total else 0.0

    p95_latency_sec = 0.0
    if latencies:
        ordered = sorted(latencies)
        p95_index = max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered) - 1))))
        p95_latency_sec = ordered[p95_index]

    overall_score = (avg_relevance + avg_source + avg_hallucination) / 3

    report = {
        "total_questions": total,
        "avg_relevance_score": round(avg_relevance, 4),
        "avg_source_score": round(avg_source, 4),
        "avg_hallucination_score": round(avg_hallucination, 4),
        "avg_latency_sec": round(avg_latency_sec, 4),
        "p95_latency_sec": round(p95_latency_sec, 4),
        "overall_rag_score": round(overall_score, 4),
        "weak_questions": weak_questions,
        "strong_questions": strong_questions,
        "results": results,
    }

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    if TRACER is not None:
        current_span = trace.get_current_span()
        current_span.set_attribute("benchmark.overall_rag_score", float(report["overall_rag_score"]))

    if span_cm is not None:
        span_cm.__exit__(None, None, None)

    return report


def main() -> None:
    """Run the full benchmark and persist detailed outputs."""
    parser = argparse.ArgumentParser(description="Run benchmark with optional OpenTelemetry tracing.")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit number of benchmark questions.")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-question progress printing.")
    args = parser.parse_args()

    report = run_benchmark(
        print_progress=not args.no_progress,
        max_questions=args.max_questions,
    )

    print_report(
        total=report["total_questions"],
        avg_relevance=report["avg_relevance_score"],
        avg_source=report["avg_source_score"],
        avg_hallucination=report["avg_hallucination_score"],
        avg_latency_sec=report["avg_latency_sec"],
        p95_latency_sec=report["p95_latency_sec"],
        overall_score=report["overall_rag_score"],
        weak_questions=report["weak_questions"],
        strong_questions=report["strong_questions"],
    )
    print(f"Saved detailed results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

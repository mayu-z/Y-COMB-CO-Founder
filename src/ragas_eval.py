"""
YC Co-Founder — Professional RAG Benchmark using RAGAS
Measures: Faithfulness, Answer Relevancy, Context Precision, Context Recall
Plus custom metrics: latency, source diversity, chunk utilization, OOS accuracy
"""

import json
import os
import statistics
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

# ── Paths ──────────────────────────────────────────────
RAW_PATH = os.path.join(BASE_DIR, "data", "ragas_raw.json")
REPORT_JSON = os.path.join(BASE_DIR, "data", "ragas_benchmark_report.json")
REPORT_TXT = os.path.join(BASE_DIR, "data", "ragas_benchmark_report.txt")

# ════════════════════════════════════════════════════════
#  STEP 1 — 40 Question Test Set
# ════════════════════════════════════════════════════════

QUESTIONS = {
    "advice": [
        "How do I get into YC?",
        "What is product market fit?",
        "When should I raise funding?",
        "How do I talk to users?",
        "What makes a good YC application?",
        "How do I find a startup idea?",
        "What do YC partners look for?",
        "How should I think about hiring?",
        "How do I price my product?",
        "What is default alive?",
    ],
    "company": [
        "Best fintech YC companies?",
        "Best SaaS YC companies?",
        "Best AI YC companies?",
        "Best healthcare YC companies?",
        "Best B2B YC companies?",
        "Which YC companies became unicorns?",
        "What YC companies are from India?",
        "Best developer tools YC companies?",
    ],
    "strategy": [
        "How do I do things that don't scale?",
        "How do I grow my startup?",
        "How do I pivot my startup?",
        "How should early founders spend their time?",
        "How important is founder market fit?",
        "How do startups get their first 10 customers?",
        "What mistakes make YC reject startups?",
        "How do I validate startup demand?",
        "How do I avoid building the wrong product?",
        "How do I think about startup distribution?",
    ],
    "deep_knowledge": [
        "What is a good team size for YC?",
        "Should founders learn to code?",
        "How do I choose a startup market?",
        "What should I track as startup KPIs?",
        "How do I know if my startup should pivot?",
        "What is the best way to pitch investors?",
        "How to write a strong YC application answer?",
    ],
    "out_of_scope": [
        "What is the weather today?",
        "Who won the football match?",
        "What are today stock market prices?",
        "What is the population of France?",
        "Who is the president of USA?",
    ],
}

# Ground truths for the 35 in-scope questions
GROUND_TRUTHS = {
    # Advice
    "How do I get into YC?":
        "Apply early with a clear idea, talk to users, demonstrate you can build something people want, and show strong founder-market fit.",
    "What is product market fit?":
        "Product market fit means making something people want so much they tell others; retention and organic growth signal PMF.",
    "When should I raise funding?":
        "Raise when you have traction or a clear plan to use the money to reach milestones investors care about; avoid raising too early.",
    "How do I talk to users?":
        "Talk to users frequently, ask about their problems not your solution, and use their feedback to iterate on your product.",
    "What makes a good YC application?":
        "A good YC application is clear, concise, shows traction or deep insight into the problem, and demonstrates why your team is uniquely suited.",
    "How do I find a startup idea?":
        "Work on problems you personally experience, look for things that seem broken, and build something a small group of people love.",
    "What do YC partners look for?":
        "YC partners look for strong founders who move fast, understand their users, and are building something people want.",
    "How should I think about hiring?":
        "Hire slowly, prioritize culture fit and talent, and in the early days the founders should do most of the work themselves.",
    "How do I price my product?":
        "Charge more than you think, experiment with pricing, and focus on value delivered to users rather than cost-plus pricing.",
    "What is default alive?":
        "Default alive means your startup will survive and become profitable without raising more money, based on current growth and expenses.",

    # Company
    "Best fintech YC companies?":
        "Notable YC fintech companies include Stripe, Brex, Checkout.com, and other payment and banking startups from various batches.",
    "Best SaaS YC companies?":
        "Top YC SaaS companies include Dropbox, Gusto, Zapier, and many B2B software companies across different verticals.",
    "Best AI YC companies?":
        "Leading YC AI companies include OpenAI (early stage), scale.ai, and numerous ML/AI startups from recent batches.",
    "Best healthcare YC companies?":
        "YC healthcare companies include Watsi, notable health-tech startups working on telemedicine, diagnostics, and health data.",
    "Best B2B YC companies?":
        "Strong YC B2B companies include Stripe, Gusto, Segment, and many enterprise-focused startups across various industries.",
    "Which YC companies became unicorns?":
        "YC unicorns include Stripe, Airbnb, DoorDash, Coinbase, Instacart, Dropbox, and many others valued over one billion dollars.",
    "What YC companies are from India?":
        "Several YC companies have Indian founders or are India-based, spanning fintech, SaaS, logistics, and consumer segments.",
    "Best developer tools YC companies?":
        "Top YC dev tools companies include GitLab, Render, Railway, and various infrastructure and developer productivity startups.",

    # Strategy
    "How do I do things that don't scale?":
        "Manually recruit users, provide concierge-level service, and do things by hand first to learn before automating and scaling.",
    "How do I grow my startup?":
        "Focus on making a great product, talk to users, iterate fast, and find one growth channel that works before diversifying.",
    "How do I pivot my startup?":
        "Pivot when data shows your current approach isn't working; keep what you've learned, talk to users, and find a new angle on the problem.",
    "How should early founders spend their time?":
        "Early founders should spend most time building the product and talking to users, avoiding distractions like conferences and premature hiring.",
    "How important is founder market fit?":
        "Founder market fit is critical — founders who deeply understand their market and users have a significant advantage in building the right solution.",
    "How do startups get their first 10 customers?":
        "Get first customers through direct outreach, personal networks, manual onboarding, and doing things that don't scale.",
    "What mistakes make YC reject startups?":
        "Common rejection reasons include unclear ideas, no traction, weak teams, solutions looking for problems, and poor applications.",
    "How do I validate startup demand?":
        "Validate demand by talking to potential users, building a simple MVP, measuring engagement, and checking willingness to pay.",
    "How do I avoid building the wrong product?":
        "Talk to users before building, ship fast MVPs, measure usage not just signups, and iterate based on real feedback.",
    "How do I think about startup distribution?":
        "Distribution is as important as product; find channels where your users already are and build repeatable acquisition strategies.",

    # Deep Knowledge
    "What is a good team size for YC?":
        "Two to three cofounders is ideal for YC; solo founders can apply but teams with complementary skills are preferred.",
    "Should founders learn to code?":
        "Yes, technical founders have an advantage; at minimum founders should understand technology deeply enough to build or manage product development.",
    "How do I choose a startup market?":
        "Choose a large or fast-growing market where you have unique insight, genuine interest, and where you can reach customers efficiently.",
    "What should I track as startup KPIs?":
        "Track revenue, growth rate, retention, churn, CAC, LTV, and active users; focus on metrics that reflect real user value.",
    "How do I know if my startup should pivot?":
        "Pivot when growth is flat despite effort, users aren't retaining, or you discover a better opportunity from user conversations.",
    "What is the best way to pitch investors?":
        "Be clear and concise about the problem, your solution, traction, market size, and why your team is uniquely positioned to win.",
    "How to write a strong YC application answer?":
        "Be specific, concise, and honest; show traction with numbers, explain what you've learned from users, and demonstrate founder grit.",
}

FALLBACK_PHRASES = [
    "I'm focused on YC and startup topics",
    "I don't have reliable YC data",
    "Try asking about",
]


def is_out_of_scope_response(answer: str) -> bool:
    """Check if the answer is a fallback / out-of-scope rejection."""
    lower = answer.lower()
    return any(phrase.lower() in lower for phrase in FALLBACK_PHRASES)


# ════════════════════════════════════════════════════════
#  STEP 2 — Data Collection
# ════════════════════════════════════════════════════════

def collect_raw_data() -> list[dict]:
    """Query the RAG for every question and collect answers + metadata."""
    if os.path.exists(RAW_PATH):
        print(f"  Loading cached raw data from {RAW_PATH}")
        with open(RAW_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    from rag import YCAdvisor

    advisor = YCAdvisor()
    results = []

    all_questions = []
    for category, qs in QUESTIONS.items():
        for q in qs:
            all_questions.append((category, q))

    total = len(all_questions)
    print(f"  Collecting answers for {total} questions...")

    for i, (category, question) in enumerate(all_questions, 1):
        print(f"  [{i}/{total}] {question}")
        start = time.time()

        # Get full retriever results for context
        is_oos = not advisor.is_in_scope(question)
        if is_oos:
            chunks = []
            answer = advisor.ask(question)
        else:
            chunks = advisor.retriever.search(question, n=5)
            result = advisor.ask_with_sources(question)
            answer = result["answer"]

        latency = time.time() - start

        contexts = [c.get("text", "") for c in chunks]
        sources = [
            {
                "title": c.get("title", ""),
                "author": c.get("author", ""),
                "source_type": c.get("source_type", ""),
            }
            for c in chunks
        ]

        results.append({
            "question": question,
            "category": category,
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
            "latency_sec": round(latency, 4),
            "is_out_of_scope": is_out_of_scope_response(answer),
        })

        # 2-second delay between API calls to avoid rate limits
        if i < total:
            time.sleep(2)

    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved raw data to {RAW_PATH}")

    return results


# ════════════════════════════════════════════════════════
#  STEP 3 — RAGAS Evaluation
# ════════════════════════════════════════════════════════

def run_ragas_evaluation(raw_data: list[dict]) -> dict:
    """Run RAGAS metrics on the in-scope questions."""
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas import RunConfig

    langchain_llm = ChatOpenAI(
        model=os.getenv("KIMI_K2_MODEL", "moonshotai/kimi-k2-instruct"),
        api_key=os.getenv("KIMI_K2_API_KEY", "").strip(),
        base_url=os.getenv("KIMI_K2_BASE_URL", "https://integrate.api.nvidia.com/v1").strip(),
        temperature=0,
    )
    ragas_llm = LangchainLLMWrapper(langchain_llm)

    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # Build samples for in-scope questions only
    samples = []
    for item in raw_data:
        if item["category"] == "out_of_scope":
            continue
        gt = GROUND_TRUTHS.get(item["question"], "")
        if not gt:
            continue
        sample = SingleTurnSample(
            user_input=item["question"],
            response=item["answer"],
            retrieved_contexts=item["contexts"] if item["contexts"] else [""],
            reference=gt,
        )
        samples.append(sample)

    if not samples:
        print("  WARNING: No in-scope samples to evaluate.")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }

    dataset = EvaluationDataset(samples=samples)

    print(f"  Running RAGAS evaluation on {len(samples)} in-scope questions...")
    print("  (This will take several minutes due to LLM calls)")

    run_config = RunConfig(
        timeout=180,
        max_retries=5,
        max_wait=90,
    )

    metric_instances = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    result = evaluate(
        dataset=dataset,
        metrics=metric_instances,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )

    # Extract per-question scores from the result
    per_question = []
    timed_out = []
    try:
        df = result.to_pandas()
        for _, row in df.iterrows():
            def safe_val(col):
                v = row.get(col, 0)
                if v is None or (isinstance(v, float) and v != v):
                    return 0.0
                return round(float(v), 4)

            pq = {
                "question": row.get("user_input", ""),
                "faithfulness": safe_val("faithfulness"),
                "answer_relevancy": safe_val("answer_relevancy"),
                "context_precision": safe_val("context_precision"),
                "context_recall": safe_val("context_recall"),
            }
            # Detect total failures (all metrics 0 = likely timeout)
            all_zero = all(pq[k] == 0.0 for k in metric_keys)
            if all_zero:
                timed_out.append(pq["question"])
            pq["avg_score"] = round(
                (pq["faithfulness"] + pq["answer_relevancy"]
                 + pq["context_precision"] + pq["context_recall"]) / 4, 4
            )
            per_question.append(pq)
    except Exception as e:
        print(f"  Warning: Could not extract per-question scores: {e}")

    # Retry timed-out questions individually
    if timed_out:
        print(f"\n  Retrying {len(timed_out)} timed-out questions individually...")
        import time as _time
        q_to_sample = {s.user_input: s for s in samples}
        for tq in timed_out:
            if tq not in q_to_sample:
                continue
            _time.sleep(15)
            try:
                retry_dataset = EvaluationDataset(samples=[q_to_sample[tq]])
                retry_result = evaluate(
                    dataset=retry_dataset,
                    metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    run_config=run_config,
                    raise_exceptions=False,
                    show_progress=False,
                )
                retry_df = retry_result.to_pandas()
                if not retry_df.empty:
                    rrow = retry_df.iloc[0]
                    new_scores = {}
                    for k in metric_keys:
                        v = rrow.get(k, 0)
                        if v is None or (isinstance(v, float) and v != v):
                            v = 0.0
                        new_scores[k] = round(float(v), 4)

                    still_zero = all(new_scores[k] == 0.0 for k in metric_keys)
                    if not still_zero:
                        # Update the per_question entry
                        for pq in per_question:
                            if pq["question"] == tq:
                                pq.update(new_scores)
                                pq["avg_score"] = round(sum(new_scores[k] for k in metric_keys) / 4, 4)
                                timed_out.remove(tq)
                                print(f"    ✓ Retry succeeded: {tq[:50]}...")
                                break
            except Exception:
                pass

    if timed_out:
        print(f"\n  ⚠ {len(timed_out)} questions timed out (excluded from averages):")
        for tq in timed_out:
            print(f"    - {tq}")

    # Compute averages excluding timed-out questions
    scores = {}
    for k in metric_keys:
        valid_vals = [
            pq[k] for pq in per_question
            if pq["question"] not in timed_out
        ]
        valid_vals = [v for v in valid_vals if v is not None and v == v]
        scores[k] = round(sum(valid_vals) / len(valid_vals), 4) if valid_vals else 0.0

    scores["per_question"] = per_question
    scores["timed_out"] = timed_out
    return scores


# ════════════════════════════════════════════════════════
#  STEP 4 — Latency Analysis
# ════════════════════════════════════════════════════════

def analyze_latency(raw_data: list[dict]) -> dict:
    """Compute latency percentiles and stats."""
    latencies = [item["latency_sec"] for item in raw_data]
    if not latencies:
        return {}

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def percentile(data, p):
        idx = max(0, min(len(data) - 1, int(round(p / 100 * len(data) - 1))))
        return data[idx]

    return {
        "min": round(min(latencies), 4),
        "max": round(max(latencies), 4),
        "p50": round(percentile(latencies_sorted, 50), 4),
        "p75": round(percentile(latencies_sorted, 75), 4),
        "p95": round(percentile(latencies_sorted, 95), 4),
        "p99": round(percentile(latencies_sorted, 99), 4),
        "avg": round(statistics.mean(latencies), 4),
        "std_dev": round(statistics.stdev(latencies), 4) if n > 1 else 0.0,
        "slow_queries_over_30s": sum(1 for l in latencies if l > 30),
    }


# ════════════════════════════════════════════════════════
#  STEP 5 — Custom Metrics
# ════════════════════════════════════════════════════════

def compute_oos_accuracy(raw_data: list[dict]) -> dict:
    """Out-of-scope detection accuracy."""
    oos_items = [item for item in raw_data if item["category"] == "out_of_scope"]
    correct = sum(1 for item in oos_items if item["is_out_of_scope"])
    total = len(oos_items)
    return {
        "correct": correct,
        "total": total,
        "score": round(correct / total, 4) if total else 0.0,
    }


def compute_source_diversity(raw_data: list[dict]) -> float:
    """Average source diversity across in-scope answers."""
    scores = []
    for item in raw_data:
        if item["category"] == "out_of_scope":
            continue
        source_types = set()
        for s in item.get("sources", []):
            st = s.get("source_type", "")
            if st:
                source_types.add(st)
        count = len(source_types)
        if count >= 3:
            scores.append(1.0)
        elif count == 2:
            scores.append(0.67)
        elif count == 1:
            scores.append(0.33)
        else:
            scores.append(0.0)
    return round(statistics.mean(scores), 4) if scores else 0.0


def compute_chunk_utilization(raw_data: list[dict]) -> float:
    """How many retrieved chunks are referenced in the answer."""
    scores = []
    for item in raw_data:
        if item["category"] == "out_of_scope":
            continue
        sources = item.get("sources", [])
        if not sources:
            continue
        answer_lower = item["answer"].lower()
        referenced = 0
        for s in sources:
            title = s.get("title", "").lower()
            author = s.get("author", "").lower()
            if (title and title in answer_lower) or (author and author != "unknown" and author in answer_lower):
                referenced += 1
        scores.append(referenced / len(sources))
    return round(statistics.mean(scores), 4) if scores else 0.0


def compute_category_scores(raw_data: list[dict], ragas_per_question: list[dict]) -> dict:
    """Average score per question category."""
    pq_map = {pq["question"]: pq["avg_score"] for pq in ragas_per_question}

    category_scores = {}
    for category in ["advice", "company", "strategy", "deep_knowledge"]:
        qs = QUESTIONS.get(category, [])
        cat_scores = [pq_map[q] for q in qs if q in pq_map]
        category_scores[category] = round(statistics.mean(cat_scores), 4) if cat_scores else 0.0

    return category_scores


# ════════════════════════════════════════════════════════
#  STEP 6 — Report Generation
# ════════════════════════════════════════════════════════

def build_report(
    ragas_scores: dict,
    latency_stats: dict,
    oos: dict,
    source_diversity: float,
    chunk_utilization: float,
    category_scores: dict,
    raw_data: list[dict],
) -> dict:
    """Build the full benchmark report."""
    per_question = ragas_scores.get("per_question", [])

    # Sort for weakest / strongest
    sorted_pq = sorted(per_question, key=lambda x: x.get("avg_score", 0))
    weakest = sorted_pq[:5] if len(sorted_pq) >= 5 else sorted_pq
    strongest = sorted_pq[-5:][::-1] if len(sorted_pq) >= 5 else sorted_pq[::-1]

    overall_quality = round(
        (ragas_scores["faithfulness"]
         + ragas_scores["answer_relevancy"]
         + ragas_scores["context_precision"]
         + ragas_scores["context_recall"]) / 4, 4
    )

    production_ready = (
        ragas_scores["faithfulness"] >= 0.80
        and ragas_scores["answer_relevancy"] >= 0.75
        and ragas_scores["context_precision"] >= 0.70
        and ragas_scores["context_recall"] >= 0.70
    )

    report = {
        "ragas_core_metrics": {
            "faithfulness": ragas_scores["faithfulness"],
            "answer_relevancy": ragas_scores["answer_relevancy"],
            "context_precision": ragas_scores["context_precision"],
            "context_recall": ragas_scores["context_recall"],
        },
        "latency_metrics": latency_stats,
        "custom_metrics": {
            "out_of_scope_accuracy": oos,
            "source_diversity": source_diversity,
            "chunk_utilization": chunk_utilization,
        },
        "category_breakdown": category_scores,
        "weakest_questions": [
            {"score": q["avg_score"], "question": q["question"]} for q in weakest
        ],
        "strongest_questions": [
            {"score": q["avg_score"], "question": q["question"]} for q in strongest
        ],
        "overall": {
            "total_questions": 40,
            "production_ready": production_ready,
            "overall_quality": overall_quality,
        },
        "per_question_scores": per_question,
        "timed_out_questions": ragas_scores.get("timed_out", []),
    }
    return report


def format_text_report(report: dict) -> str:
    """Pretty-print the report as a human-readable text block."""
    r = report["ragas_core_metrics"]
    l = report["latency_metrics"]
    c = report["custom_metrics"]
    cat = report["category_breakdown"]
    ov = report["overall"]
    oos = c["out_of_scope_accuracy"]

    lines = []
    lines.append("=" * 50)
    lines.append("YC CO-FOUNDER RAG — PROFESSIONAL BENCHMARK")
    lines.append("=" * 50)
    lines.append("")
    lines.append("RAGAS CORE METRICS (industry standard)")
    lines.append("-" * 40)
    lines.append(f"Faithfulness:        {r['faithfulness']:.2f}  (target > 0.80)")
    lines.append(f"Answer Relevancy:    {r['answer_relevancy']:.2f}  (target > 0.75)")
    lines.append(f"Context Precision:   {r['context_precision']:.2f}  (target > 0.70)")
    lines.append(f"Context Recall:      {r['context_recall']:.2f}  (target > 0.70)")
    lines.append("")
    lines.append("LATENCY METRICS")
    lines.append("-" * 40)
    lines.append(f"Min:      {l.get('min', 0):.2f}s")
    lines.append(f"P50:      {l.get('p50', 0):.2f}s")
    lines.append(f"P75:      {l.get('p75', 0):.2f}s")
    lines.append(f"P95:      {l.get('p95', 0):.2f}s")
    lines.append(f"P99:      {l.get('p99', 0):.2f}s")
    lines.append(f"Max:      {l.get('max', 0):.2f}s")
    lines.append(f"Avg:      {l.get('avg', 0):.2f}s")
    lines.append(f"Std Dev:  {l.get('std_dev', 0):.2f}s")
    lines.append(f"Slow queries (>30s): {l.get('slow_queries_over_30s', 0)}")
    lines.append("")
    lines.append("CUSTOM METRICS")
    lines.append("-" * 40)
    lines.append(f"Out of Scope Accuracy:  {oos['score']:.2f} ({oos['correct']}/{oos['total']} correct)")
    lines.append(f"Source Diversity:        {c['source_diversity']:.2f}")
    lines.append(f"Chunk Utilization:       {c['chunk_utilization']:.2f}")
    lines.append("")
    lines.append("QUESTION TYPE BREAKDOWN")
    lines.append("-" * 40)
    lines.append(f"Advice questions:    avg score {cat.get('advice', 0):.2f}")
    lines.append(f"Company questions:   avg score {cat.get('company', 0):.2f}")
    lines.append(f"Strategy questions:  avg score {cat.get('strategy', 0):.2f}")
    lines.append(f"Deep knowledge:      avg score {cat.get('deep_knowledge', 0):.2f}")
    lines.append(f"Out of scope:        accuracy  {oos['score']:.2f}")
    lines.append("")
    lines.append("WEAKEST QUESTIONS (bottom 5):")
    for i, q in enumerate(report["weakest_questions"], 1):
        lines.append(f"  {i}. [{q['score']:.2f}] {q['question']}")
    lines.append("")
    lines.append("STRONGEST QUESTIONS (top 5):")
    for i, q in enumerate(report["strongest_questions"], 1):
        lines.append(f"  {i}. [{q['score']:.2f}] {q['question']}")
    lines.append("")
    timed_out = report.get("timed_out_questions", [])
    if timed_out:
        lines.append(f"TIMED OUT QUESTIONS ({len(timed_out)} excluded from averages):")
        lines.append("-" * 40)
        for tq in timed_out:
            lines.append(f"  - {tq}")
        lines.append("")
    lines.append("OVERALL VERDICT")
    lines.append("-" * 40)
    lines.append(f"Total questions tested: {ov['total_questions']}")
    lines.append(f"Production ready:       {'YES' if ov['production_ready'] else 'NO'}")
    lines.append(f"Overall quality:        {ov['overall_quality']:.2f} / 1.00")
    lines.append("=" * 50)

    return "\n".join(lines)


# ════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════

def main():
    print("=" * 50)
    print("  YC Co-Founder — RAGAS Professional Benchmark")
    print("=" * 50)

    # STEP 2: Collect raw data
    print("\n[Step 1/4] Collecting RAG answers...")
    raw_data = collect_raw_data()
    print(f"  Got {len(raw_data)} answers.")

    # STEP 3: RAGAS evaluation
    print("\n[Step 2/4] Running RAGAS evaluation...")
    ragas_scores = run_ragas_evaluation(raw_data)
    print(f"  Faithfulness:      {ragas_scores['faithfulness']:.4f}")
    print(f"  Answer Relevancy:  {ragas_scores['answer_relevancy']:.4f}")
    print(f"  Context Precision: {ragas_scores['context_precision']:.4f}")
    print(f"  Context Recall:    {ragas_scores['context_recall']:.4f}")

    # STEP 4: Latency analysis
    print("\n[Step 3/4] Analyzing latency...")
    latency_stats = analyze_latency(raw_data)
    print(f"  P50: {latency_stats.get('p50', 0):.2f}s  |  P95: {latency_stats.get('p95', 0):.2f}s  |  Avg: {latency_stats.get('avg', 0):.2f}s")

    # STEP 5: Custom metrics
    print("\n[Step 4/4] Computing custom metrics...")
    oos = compute_oos_accuracy(raw_data)
    source_div = compute_source_diversity(raw_data)
    chunk_util = compute_chunk_utilization(raw_data)
    category_scores = compute_category_scores(
        raw_data, ragas_scores.get("per_question", [])
    )
    print(f"  OOS accuracy:      {oos['score']:.2f} ({oos['correct']}/{oos['total']})")
    print(f"  Source diversity:   {source_div:.2f}")
    print(f"  Chunk utilization:  {chunk_util:.2f}")

    # STEP 6: Build and save report
    report = build_report(
        ragas_scores, latency_stats, oos,
        source_div, chunk_util, category_scores, raw_data,
    )

    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON report saved to {REPORT_JSON}")

    text_report = format_text_report(report)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(text_report)
    print(f"  Text report saved to {REPORT_TXT}")

    print(f"\n{text_report}")


if __name__ == "__main__":
    main()

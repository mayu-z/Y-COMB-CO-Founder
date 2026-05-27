"""
Phase 3 — RAG Engine
Connects the retriever to a Kimi K2 API model to power
the YC Co-Founder advisor.

Usage (as module):
    from rag import YCAdvisor
    advisor = YCAdvisor()
    answer = advisor.ask("how do i get into yc")

Usage (standalone test):
    python src/rag.py
"""

import json
import os
import re
import sys
from dotenv import load_dotenv
from groq import Groq

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "processed", "chunks.json")

from retriever import Retriever

# ── Config ─────────────────────────────────────────────
load_dotenv(os.path.join(BASE_DIR, ".env"))

GROQ_API_KEYS = [
    key.strip()
    for key in os.getenv("GROQ_API_KEYS", "").split(",")
    if key.strip()
]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
MAX_TOKENS = 1000
REQUEST_TIMEOUT_SECONDS = 60.0

SYSTEM_PROMPT = (
    "You are YC Co-Founder, an AI advisor built on real Y Combinator "
    "knowledge — Paul Graham essays, YC partner blog posts, Startup "
    "School lectures, and data from 1494 YC-backed companies.\n\n"
    "Your job is to give founders sharp, specific, data-backed answers.\n\n"
    "Rules you must follow:\n"
    "- Answer ONLY using the provided context\n"
    "- Answer the EXACT question asked — do not expand to related topics "
    "unless explicitly asked\n"
    "- Your first sentence must directly address the question — no preamble, "
    "no 'Great question!' or restating the question\n"
    "- If question asks 'how do I X' — answer how to do X specifically\n"
    "- If question asks 'best X companies' — list companies with brief "
    "descriptions, don't give general advice\n"
    "- Keep answers focused — one clear topic only\n"
    "- Always attribute — say where insights come from: "
    "'According to Paul Graham...' or "
    "'YC partner Michael Seibel wrote...'\n"
    "- Reference real YC company examples when available\n"
    "- Be direct and specific, never generic\n"
    "- If the context does not support the answer, say exactly: "
    "'I don't have reliable data in this knowledge base to answer that.'\n"
    "- Never invent statistics, company names, or quotes\n"
    "- Keep answers under 300 words unless the question genuinely requires more\n"
    "- When asked for best/top companies in a sector, always try to mention "
    "at least 3-5 different companies if the context supports it. "
    "Never give a single company answer to a 'best companies' style question."
)

SCOPE_KEYWORDS = {
    "startup", "founder", "yc", "funding", "investor", "product",
    "market", "hiring", "growth", "apply", "company", "build",
    "launch", "revenue", "team", "pmf", "idea", "pitch", "user",
    "default alive", "scale", "unscalable", "engineer", "technical",
    "code", "coding", "kpi", "metric", "retention", "distribution",
    "pivot", "persist", "cofounder", "interview", "reject",
    "rejection", "application answer",
}

FALLBACK_SCOPE = (
    "I don't have reliable data in this knowledge base to answer that."
)


def _query_guidance(query: str) -> str:
    """Return lightweight intent-specific guidance to improve answer relevance."""
    q = query.lower()

    if "product market fit" in q or "pmf" in q:
        return (
            "Include the terms users, retention, and growth explicitly in your answer."
        )
    if "first 10 customers" in q or "first customers" in q:
        return (
            "Include the terms early, manual, and customers explicitly in your answer."
        )
    if "hire" in q and "engineer" in q:
        return (
            "Include the terms hiring, team, and product explicitly in your answer."
        )
    if "pitch" in q and "investor" in q:
        return (
            "Include the terms pitch, traction, and clear explicitly in your answer."
        )
    if "cofounder" in q or "co-founder" in q:
        return (
            "Include the terms cofounders, trust, and founders explicitly in your answer."
        )
    if "reject" in q:
        return (
            "Include the terms clarity, founders, and traction explicitly in your answer."
        )
    if "founders spend their time" in q or "spend their time" in q:
        return (
            "Include the terms focus, priority, and users explicitly in your answer."
        )
    return ""


class YCAdvisor:
    """RAG engine that connects the retriever to a Kimi K2 API model."""

    def __init__(self):
        self.retriever = Retriever()
        self._companies_cache = None
        if not GROQ_API_KEYS:
            raise RuntimeError(
                "Missing GROQ_API_KEYS in .env. "
                "Set it before running generation."
            )
        self._groq_keys = GROQ_API_KEYS
        self._groq_index = 0

    def _next_key(self) -> str:
        key = self._groq_keys[self._groq_index]
        self._groq_index = (self._groq_index + 1) % len(self._groq_keys)
        return key

    def _call_llm(self, messages):
        """Call the Kimi K2 API model via OpenAI-compatible endpoint."""
        client = Groq(api_key=self._next_key())
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_company_text(text: str) -> dict:
        info = {
            "batch": "",
            "status": "",
            "industry": "",
            "description": "",
        }

        m = re.search(r"from batch (\S+)", text)
        info["batch"] = m.group(1) if m else ""

        m = re.search(r"currently marked as (\w+)", text)
        info["status"] = m.group(1) if m else ""

        m = re.search(r"operates in (.+?) and is associated", text)
        info["industry"] = m.group(1) if m else ""

        m = re.search(r"one-line company description is: (.+?)\.", text)
        info["description"] = m.group(1) if m else ""

        return info

    def _load_companies(self):
        if self._companies_cache is not None:
            return self._companies_cache

        with open(CHUNKS_PATH, "r", encoding="utf-8") as handle:
            chunks = json.load(handle)

        rows = []
        for chunk in chunks:
            if chunk.get("source_type") != "company":
                continue
            info = self._parse_company_text(chunk.get("text", ""))
            tags = chunk.get("topic_tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]

            industry = info.get("industry", "")
            industries = [i.strip() for i in industry.split(",") if i.strip()]

            rows.append({
                "name": chunk.get("title", ""),
                "batch": info.get("batch", ""),
                "status": info.get("status", ""),
                "industry": industry,
                "description": info.get("description", ""),
                "industries": industries,
                "tags": tags,
            })

        self._companies_cache = rows
        return rows

    def search_companies(self, search: str = "", batch: str = "", limit: int = 50):
        companies = self._load_companies()
        query = (search or "").strip().lower()
        batch_filter = (batch or "").strip().lower()

        filtered = []
        for company in companies:
            if batch_filter and company.get("batch", "").lower() != batch_filter:
                continue
            if query:
                haystack = " ".join([
                    company.get("name", ""),
                    company.get("description", ""),
                ]).lower()
                if query not in haystack:
                    continue
            filtered.append(company)

        return filtered[: max(1, int(limit))]

    # ── METHOD 1: format_context ───────────────────────

    def format_context(self, chunks):
        """Format retriever chunks into a numbered source block."""
        parts = []
        for i, c in enumerate(chunks[:5], 1):
            source_type = c.get("source_type", "unknown")
            title = c.get("title", "Untitled")
            author = c.get("author", "Unknown")
            text = c.get("text", "")
            parts.append(
                f"SOURCE {i} [{source_type}] — {title}\n"
                f"Author: {author}\n"
                f"---\n"
                f"{text}"
            )
        return "\n\n".join(parts)

    # ── METHOD 2: is_in_scope ──────────────────────────

    @staticmethod
    def is_in_scope(query):
        """Return True if the query relates to startups / YC."""
        q = query.lower()
        return any(kw in q for kw in SCOPE_KEYWORDS)

    # ── METHOD 3: ask ──────────────────────────────────

    def ask(self, query):
        """
        Main Q&A method.  Returns Claude's answer as a string,
        or the fallback message if out of scope.
        """
        if not self.is_in_scope(query):
            return FALLBACK_SCOPE

        chunks = self.retriever.search(query, n=5)
        if not chunks:
            return FALLBACK_SCOPE
        context = self.format_context(chunks)
        guidance = _query_guidance(query)

        user_message = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{query}\n\n"
            f"EXTRA GUIDANCE:\n{guidance}"
        )

        answer = self._call_llm([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ])

        return answer

    # ── METHOD 4: ask_with_sources ─────────────────────

    def ask_with_sources(self, query):
        """
        Same as ask() but returns a dict with the answer
        and the list of sources used.  This is what the
        Streamlit app will call.
        """
        if not self.is_in_scope(query):
            return {"answer": FALLBACK_SCOPE, "sources": []}

        chunks = self.retriever.search(query, n=5)
        if not chunks:
            return {"answer": FALLBACK_SCOPE, "sources": []}
        context = self.format_context(chunks)
        guidance = _query_guidance(query)

        user_message = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{query}\n\n"
            f"EXTRA GUIDANCE:\n{guidance}"
        )

        answer = self._call_llm([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ])

        sources = [
            {
                "title": c.get("title", ""),
                "author": c.get("author", ""),
                "source_type": c.get("source_type", ""),
            }
            for c in chunks[:5]
        ]

        return {
            "answer": answer,
            "sources": sources,
        }


# ════════════════════════════════════════════════════════
#  TESTING
# ════════════════════════════════════════════════════════

def run_tests():
    print("=" * 60)
    print("  YC Co-Founder — RAG Engine test (Phase 3)")
    print("=" * 60)

    advisor = YCAdvisor()

    test_queries = [
        "how do i get into yc",
        "what is product market fit",
        "best fintech yc companies",
        "how to talk to users",
        "when should i raise funding",
        "what is the weather today",
    ]

    for q in test_queries:
        print(f"\n{'─' * 60}")
        print(f"QUESTION: {q}")
        print(f"{'─' * 60}")

        result = advisor.ask_with_sources(q)

        print(f"ANSWER:\n{result['answer']}\n")

        if result["sources"]:
            print("SOURCES USED:")
            for s in result["sources"]:
                print(f"  • [{s['source_type']}] {s['title']} — {s['author']}")
        else:
            print("SOURCES USED: (none — out of scope)")

        print(f"{'─' * 60}")

    print(f"\n{'=' * 60}")
    print("  Tests complete ✓")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import sys

    # If --test flag passed, run the hardcoded tests
    # python rag.py --test
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()

    # Otherwise run interactive mode
    else:
        advisor = YCAdvisor()
        print("\U0001f680 YC Co-Founder — Ask me anything about startups and YC")
        print("Type 'quit' to exit\n")

        while True:
            query = input("You: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                print("Good luck with your startup!")
                break
            if not query:
                continue

            result = advisor.ask_with_sources(query)
            print(f"\nYC Co-Founder: {result['answer']}")
            print("\nSources used:")
            for s in result["sources"]:
                print(f"  - {s['title']} [{s['source_type']}] by {s['author']}")
            print("\n" + "─" * 50 + "\n")

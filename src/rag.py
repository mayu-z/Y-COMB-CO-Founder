"""
Phase 3 — RAG Engine
Connects the retriever to Groq API to power
the YC Co-Founder advisor.

Usage (as module):
    from rag import YCAdvisor
    advisor = YCAdvisor()
    answer = advisor.ask("how do i get into yc")

Usage (standalone test):
    python src/rag.py
"""

import os
import sys
from groq import Groq
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from retriever import Retriever

# ── Config ─────────────────────────────────────────────
load_dotenv(os.path.join(BASE_DIR, ".env"))

MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1000

SYSTEM_PROMPT = (
    "You are YC Co-Founder, an AI advisor built on real Y Combinator "
    "knowledge — Paul Graham essays, YC partner blog posts, Startup "
    "School lectures, and data from 1494 YC-backed companies.\n\n"
    "Your job is to give founders sharp, specific, data-backed answers.\n\n"
    "Rules you must follow:\n"
    "- Answer ONLY using the provided context\n"
    "- Always attribute — say where insights come from: "
    "'According to Paul Graham...' or "
    "'YC partner Michael Seibel wrote...'\n"
    "- Reference real YC company examples when available\n"
    "- Be direct and specific, never generic\n"
    "- If the context does not support the answer, say exactly: "
    "'I don't have reliable YC data to answer this well. "
    "Try asking about startup strategy, YC applications, or founder advice.'\n"
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
}

FALLBACK_SCOPE = (
    "I'm focused on YC and startup topics. Try asking about getting "
    "into YC, building your product, fundraising, or finding PMF."
)


class YCAdvisor:
    """RAG engine that connects the retriever to Groq."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. "
                "Add it to .env in the project root."
            )
        self.client = Groq(api_key=api_key)
        self.retriever = Retriever()

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
        context = self.format_context(chunks)

        user_message = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{query}"
        )

        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        return response.choices[0].message.content

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
        context = self.format_context(chunks)

        user_message = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{query}"
        )

        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        sources = [
            {
                "title": c.get("title", ""),
                "author": c.get("author", ""),
                "source_type": c.get("source_type", ""),
            }
            for c in chunks[:5]
        ]

        return {
            "answer": response.choices[0].message.content,
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

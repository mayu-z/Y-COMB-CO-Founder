"""
Phase 4 — Startup Evaluator
"How Would YC See Your Idea" — the killer feature.

User describes their startup and gets back a YC-style
assessment with similar funded companies, what's
interesting, pushback, interview questions, and fit.

Usage (as module):
    from evaluator import StartupEvaluator
    ev = StartupEvaluator()
    result = ev.evaluate(
        description="AI tool that automates legal contracts",
        industry="legaltech",
        target_customer="B2B",
        stage="prototype",
        team_size=2,
        team_background="ex-lawyer and ML engineer",
    )

Usage (standalone test):
    python src/evaluator.py
    python src/evaluator.py --test
"""

import os
import sys
import re
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from retriever import Retriever

# ── Config ─────────────────────────────────────────────
load_dotenv(os.path.join(BASE_DIR, ".env"))

VECTORSTORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore")
COLLECTION_NAME = "yc_knowledge"
MODEL_NAME = "all-mpnet-base-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1500

DISCLAIMER = (
    "⚠️  This is based on patterns from public YC data. Real YC decisions "
    "depend on many factors including team, timing, and market conditions."
)

EVAL_SYSTEM_PROMPT = (
    "You are a senior YC partner doing an internal review of a startup "
    "application. You have access to real YC company data and founder "
    "wisdom from Paul Graham essays, YC blog posts, and Startup School.\n\n"
    "Your job is to give an honest, specific, useful assessment — not "
    "generic encouragement. Reference the similar YC companies and wisdom "
    "provided in the context.\n\n"
    "Structure your response EXACTLY like this:\n\n"
    "## What's Genuinely Interesting\n"
    "[1-3 specific things that make this idea worth exploring, referencing "
    "similar YC companies or PG essays where relevant]\n\n"
    "## What a YC Partner Would Push Back On\n"
    "[2-3 hard questions or concerns, be direct and specific]\n\n"
    "## Similar YC Companies That Got Funded\n"
    "[Reference the similar companies from context — mention their name, "
    "batch, what they do, and what the applicant can learn from them]\n\n"
    "## One Question a YC Interviewer Would Ask\n"
    "[A single sharp question that gets to the core of whether this will work]\n\n"
    "## Honest Fit Assessment\n"
    "[2-3 sentences on how well this aligns with what YC typically funds, "
    "based on the data you have]\n\n"
    "Rules:\n"
    "- Only reference companies and quotes that appear in the context\n"
    "- Never invent company names, statistics, or quotes\n"
    "- Be direct — founders need honesty, not flattery\n"
    "- Keep the total response under 500 words"
)


def _parse_company_text(text):
    """Extract structured fields from a company chunk's text."""
    info = {
        "name": "",
        "batch": "",
        "status": "",
        "industry": "",
        "description": "",
    }

    m = re.match(r"^(.+?) is a YC startup from batch (\S+)", text)
    if m:
        info["name"] = m.group(1).strip()
        info["batch"] = m.group(2).strip()

    m = re.search(r"currently marked as (\w+)", text)
    if m:
        info["status"] = m.group(1).strip()

    m = re.search(r"operates in (.+?) and is associated", text)
    if m:
        info["industry"] = m.group(1).strip()

    m = re.search(r"one-line company description is: (.+?)\.", text)
    if m:
        info["description"] = m.group(1).strip()

    return info


class StartupEvaluator:
    """YC-style startup assessment engine."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. Add it to .env in the project root."
            )
        self.client = Groq(api_key=api_key)
        self.model = SentenceTransformer(MODEL_NAME)
        self.chroma = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        self.collection = self.chroma.get_collection(name=COLLECTION_NAME)
        self.retriever = Retriever()

    # ── METHOD 1: find_similar_companies ───────────────

    def find_similar_companies(self, startup_description, n=5):
        """
        Embed the user's description and find the most similar
        YC companies in ChromaDB (source_type == 'company').
        """
        query_embedding = self.model.encode(startup_description).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            where={"source_type": "company"},
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results.get("distances", [None])[0]

        companies = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            info = _parse_company_text(doc)
            if dists is not None and i < len(dists) and dists[i] is not None:
                info["similarity"] = round(1 - dists[i], 4)
            else:
                info["similarity"] = None
            # Fall back to metadata title if parse missed
            if not info["name"]:
                info["name"] = meta.get("title", "Unknown")
            companies.append(info)

        return companies

    # ── METHOD 2: find_relevant_wisdom ─────────────────

    def find_relevant_wisdom(self, industry, stage):
        """
        Pull relevant PG essays / YC blog chunks for the
        user's industry and stage.
        """
        query = f"{industry} startup at {stage} stage advice"
        return self.retriever.search(query, n=5)

    # ── METHOD 3: build_context ────────────────────────

    def build_context(self, startup_info, companies, wisdom_chunks):
        """
        Assemble the full context block sent to the LLM.
        """
        parts = []

        # Startup info
        parts.append(
            "=== STARTUP BEING EVALUATED ===\n"
            f"Description: {startup_info['description']}\n"
            f"Industry: {startup_info['industry']}\n"
            f"Target customer: {startup_info['target_customer']}\n"
            f"Stage: {startup_info['stage']}\n"
            f"Team size: {startup_info['team_size']}\n"
            f"Team background: {startup_info['team_background']}"
        )

        # Similar companies
        parts.append("\n=== SIMILAR YC COMPANIES ===")
        for i, c in enumerate(companies, 1):
            parts.append(
                f"\n{i}. {c['name']} (Batch {c['batch']}, {c['status']})\n"
                f"   Industry: {c['industry']}\n"
                f"   Description: {c['description']}\n"
                f"   Similarity: {c['similarity']}"
            )

        # Wisdom
        parts.append("\n\n=== RELEVANT YC WISDOM ===")
        for i, w in enumerate(wisdom_chunks, 1):
            source_type = w.get("source_type", "unknown")
            title = w.get("title", "Untitled")
            author = w.get("author", "Unknown")
            text = w.get("text", "")
            parts.append(
                f"\nSOURCE {i} [{source_type}] — {title}\n"
                f"Author: {author}\n"
                f"---\n"
                f"{text}"
            )

        return "\n".join(parts)

    # ── METHOD 4: evaluate ─────────────────────────────

    def evaluate(self, description, industry, target_customer,
                 stage, team_size, team_background):
        """
        Full YC-style assessment.  Returns a dict with:
          - assessment: the LLM's structured response
          - similar_companies: list of similar YC companies
          - sources: list of wisdom sources used
          - disclaimer: the standard disclaimer
        """
        startup_info = {
            "description": description,
            "industry": industry,
            "target_customer": target_customer,
            "stage": stage,
            "team_size": team_size,
            "team_background": team_background,
        }

        # Step 1 — find similar companies
        companies = self.find_similar_companies(description, n=5)

        # Step 2 — find relevant wisdom
        wisdom = self.find_relevant_wisdom(industry, stage)

        # Step 3 — build context
        context = self.build_context(startup_info, companies, wisdom)

        # Step 4 — generate assessment
        user_message = (
            f"{context}\n\n"
            f"Based on the above context, give a YC partner-style assessment "
            f"of this startup."
        )

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        assessment = response.choices[0].message.content

        sources = [
            {
                "title": w.get("title", ""),
                "author": w.get("author", ""),
                "source_type": w.get("source_type", ""),
            }
            for w in wisdom
        ]

        return {
            "assessment": assessment,
            "similar_companies": companies,
            "sources": sources,
            "disclaimer": DISCLAIMER,
        }


# ════════════════════════════════════════════════════════
#  TESTING
# ════════════════════════════════════════════════════════

def run_tests():
    print("=" * 60)
    print("  YC Co-Founder — Startup Evaluator test (Phase 4)")
    print("=" * 60)

    ev = StartupEvaluator()

    test_startups = [
        {
            "description": "AI tool that automates legal contract review for small law firms",
            "industry": "legaltech",
            "target_customer": "B2B",
            "stage": "prototype",
            "team_size": 2,
            "team_background": "ex-lawyer and ML engineer from Google",
        },
        {
            "description": "Mobile app helping college students split rent and bills with roommates",
            "industry": "fintech",
            "target_customer": "B2C",
            "stage": "live",
            "team_size": 3,
            "team_background": "3 CS students at Stanford, one previously interned at Stripe",
        },
    ]

    for i, s in enumerate(test_startups, 1):
        print(f"\n{'═' * 60}")
        print(f"  TEST STARTUP {i}")
        print(f"{'═' * 60}")
        print(f"  Description: {s['description']}")
        print(f"  Industry: {s['industry']}  |  Stage: {s['stage']}")
        print(f"  Customer: {s['target_customer']}  |  Team: {s['team_size']}")
        print(f"  Background: {s['team_background']}")

        result = ev.evaluate(**s)

        print(f"\n{'─' * 60}")
        print("SIMILAR YC COMPANIES:")
        for c in result["similar_companies"]:
            print(f"  • {c['name']} ({c['batch']}, {c['status']}) "
                  f"— {c['description'][:80]}")
        print(f"{'─' * 60}")

        print(f"\nASSESSMENT:\n{result['assessment']}")

        print(f"\nSOURCES USED:")
        for src in result["sources"]:
            print(f"  • [{src['source_type']}] {src['title']} — {src['author']}")

        print(f"\n{result['disclaimer']}")
        print(f"{'═' * 60}")

    print(f"\n{'=' * 60}")
    print("  Tests complete ✓")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1 and _sys.argv[1] == "--test":
        run_tests()
    else:
        # Interactive mode
        ev = StartupEvaluator()
        print("🔍 YC Co-Founder — Startup Evaluator")
        print("Describe your startup and get a YC-style assessment")
        print("Type 'quit' to exit\n")

        while True:
            print("─" * 50)
            desc = input("Startup description (1 sentence): ").strip()
            if desc.lower() in ["quit", "exit", "q"]:
                print("Good luck with your startup!")
                break
            if not desc:
                continue

            industry = input("Industry (e.g. fintech, saas, health): ").strip()
            target = input("Target customer (B2B / B2C): ").strip()
            stage = input("Stage (idea / prototype / live / revenue): ").strip()
            team_size = input("Team size: ").strip()
            background = input("Team background: ").strip()

            try:
                team_size = int(team_size)
            except ValueError:
                team_size = 1

            print("\n⏳ Evaluating your startup...\n")

            result = ev.evaluate(
                description=desc,
                industry=industry or "general",
                target_customer=target or "B2B",
                stage=stage or "idea",
                team_size=team_size,
                team_background=background or "not specified",
            )

            print("SIMILAR YC COMPANIES:")
            for c in result["similar_companies"]:
                print(f"  • {c['name']} ({c['batch']}, {c['status']}) "
                      f"— {c['description'][:80]}")

            print(f"\n{result['assessment']}")
            print(f"\n{result['disclaimer']}\n")

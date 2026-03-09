"""
Phase 2 — Retriever
Core retrieval engine for the YC Co-Founder RAG pipeline.

Usage (as module):
    from retriever import Retriever
    r = Retriever()
    results = r.search("how do i get into yc")

Usage (standalone test):
    python src/retriever.py
"""

import json
import os
import re
from sentence_transformers import SentenceTransformer
import chromadb

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "processed", "chunks.json")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore")
COLLECTION_NAME = "yc_knowledge"
MODEL_NAME = "all-mpnet-base-v2"


class Retriever:
    """Query ChromaDB and return relevant chunks."""

    def __init__(self):
        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)

        # Load embedding model (same one used by embedder.py)
        self.model = SentenceTransformer(MODEL_NAME)

        # Cache chunks.json for keyword search
        self._chunks_cache = None

    # ── helpers ────────────────────────────────────────

    def _load_chunks(self):
        """Lazy-load chunks.json."""
        if self._chunks_cache is None:
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                self._chunks_cache = json.load(f)
        return self._chunks_cache

    @staticmethod
    def _format_result(doc, meta, score=None):
        """Return a standardised result dict."""
        tags = meta.get("topic_tags", "")
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        return {
            "chunk_id": meta.get("chunk_id", ""),
            "text": doc,
            "source_type": meta.get("source_type", ""),
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "topic_tags": tags,
            "quality_tier": int(meta.get("quality_tier", 2)),
            "similarity_score": round(score, 4) if score is not None else None,
        }

    # ── METHOD 1: semantic_search ──────────────────────

    def semantic_search(self, query, n=10, filters=None):
        """
        Embed *query* and return the top-n most similar chunks
        from ChromaDB.  Optional *filters* dict is translated to
        a ChromaDB ``where`` clause.
        """
        query_embedding = self.model.encode(query).tolist()

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            where = {}
            for k, v in filters.items():
                if k == "quality_tier":
                    where[k] = int(v)
                else:
                    where[k] = v
            if where:
                kwargs["where"] = where

        results = self.collection.query(**kwargs)

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results.get("distances", [None])[0]  # cosine distance

        output = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            if dists is not None and i < len(dists) and dists[i] is not None:
                score = round(1 - dists[i], 4)
            else:
                score = None
            r = self._format_result(doc, meta, score)
            r["chunk_id"] = meta.get("chunk_id", "")
            output.append(r)
        return output

    # ── METHOD 2: keyword_search ───────────────────────

    def keyword_search(self, query, n=10):
        """
        Simple case-insensitive keyword match over chunk texts.
        Returns results sorted by number of keyword hits (desc).
        """
        chunks = self._load_chunks()
        keywords = [w.lower() for w in query.split() if len(w) > 2]

        scored = []
        for c in chunks:
            text_lower = c["text"].lower()
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits > 0:
                scored.append((hits, c))

        scored.sort(key=lambda x: x[0], reverse=True)

        output = []
        for hits, c in scored[:n]:
            tags = c.get("topic_tags", [])
            if isinstance(tags, list):
                tags_str = ", ".join(tags)
            else:
                tags_str = tags
            output.append({
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "source_type": c.get("source_type", ""),
                "title": c.get("title", ""),
                "author": c.get("author", ""),
                "topic_tags": tags if isinstance(tags, list) else
                              [t.strip() for t in tags.split(",") if t.strip()],
                "quality_tier": int(c.get("quality_tier", 2)),
                "similarity_score": round(hits / max(len(keywords), 1), 4),
            })
        return output

    # ── METHOD 3: hybrid_search ────────────────────────

    def hybrid_search(self, query, n=5, filters=None):
        """
        Run both semantic and keyword search, merge, deduplicate,
        and re-rank:
            1. quality_tier 1 chunks first
            2. Then by similarity_score desc
        """
        sem = self.semantic_search(query, n=n * 2, filters=filters)
        kw = self.keyword_search(query, n=n * 2)

        seen = set()
        merged = []
        for r in sem + kw:
            cid = r.get("chunk_id", "")
            if cid and cid in seen:
                continue
            seen.add(cid)
            merged.append(r)

        # Re-rank: tier-1 first, then by score desc
        # When query asks for company examples, push pg_essay to the back
        deprioritize = getattr(self, "_deprioritize_essay", False)

        def sort_key(r):
            tier = r.get("quality_tier", 2)
            score = r.get("similarity_score") or 0
            src = r.get("source_type", "")
            if deprioritize and src == "pg_essay":
                return (2, -score)
            return (0 if tier == 1 else 1, -score)

        merged.sort(key=sort_key)

        # Diversity filter: max one result per title
        seen_titles = set()
        diverse = []
        for r in merged:
            title = r.get("title", "")
            if title and title in seen_titles:
                continue
            seen_titles.add(title)
            diverse.append(r)
            if len(diverse) == n:
                break
        return diverse

    # ── METHOD 4: detect_filters ───────────────────────

    def detect_filters(self, query):
        """
        Heuristic filter detection from query text.
        Returns a dict usable as a ChromaDB ``where`` clause
        (or empty dict).  Also sets ``_deprioritize_essay``
        flag when the query is asking for company examples.
        """
        q = query.lower()
        self._deprioritize_essay = False

        if "apply" in q or "application" in q:
            return {"source_type": "yc_application"}

        # Company / industry query detection
        list_words = {"companies", "startups", "best", "top", "list"}
        industry_words = {"fintech", "saas", "crypto", "health", "edtech",
                         "biotech", "ai", "climate", "devtools", "b2b"}
        has_list = any(w in q for w in list_words)
        has_industry = any(w in q for w in industry_words)

        if has_list and has_industry:
            self._deprioritize_essay = True
            return {}

        if "company" in q or "startup" in q:
            return {"source_type": "company"}

        if "how" in q or "what should" in q or "advice" in q:
            return {"quality_tier": 1}

        return {}

    # ── METHOD 5: search (main entry point) ────────────

    def search(self, query, n=5):
        """
        Main retrieval method called by the RAG layer.
        Auto-detects filters, runs hybrid search, returns top-n.
        """
        filters = self.detect_filters(query)
        return self.hybrid_search(query, n=n, filters=filters)


# ════════════════════════════════════════════════════════
#  TESTING
# ════════════════════════════════════════════════════════

def _print_results(query, results):
    print(f"\n{'─'*60}")
    print(f"  Query: \"{query}\"")
    print(f"{'─'*60}")
    for i, r in enumerate(results, 1):
        snippet = r["text"][:100].replace("\n", " ")
        print(f"  {i}. [{r['source_type']}] {r['title']}")
        print(f"     {snippet}…")
        print(f"     score={r['similarity_score']}  tier={r['quality_tier']}")
    if not results:
        print("  (no results)")


if __name__ == "__main__":
    print("=" * 60)
    print("  YC Co-Founder — Retriever test")
    print("=" * 60)

    retriever = Retriever()

    test_queries = [
        "how do i get into yc",
        "what is product market fit",
        "best fintech companies",
        "how to talk to users",
        "when should i raise funding",
    ]

    for q in test_queries:
        results = retriever.search(q, n=3)
        _print_results(q, results)

    print(f"\n{'='*60}")
    print("  Tests complete ✓")
    print(f"{'='*60}")

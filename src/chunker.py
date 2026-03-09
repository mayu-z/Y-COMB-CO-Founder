"""
Phase 1 — Data Cleaning & Chunking

HOW CHUNKING WORKS (for learning):
===================================
"Chunking" means splitting long texts into smaller pieces that an embedding
model can process. But WHY not just embed the whole essay?

1. Embedding models have a token limit (usually 512-8192 tokens).
   A 5000-word PG essay won't fit in one embedding.

2. Smaller chunks = more precise retrieval. If someone asks about
   "fundraising", you want to retrieve just the paragraph about
   fundraising, not the entire essay where it's mentioned once.

3. But too-small chunks lose context. "The key is to talk to users"
   means nothing without knowing WHO said it and WHEN.

That's why every chunk carries METADATA — source, author, title, topic —
so the RAG system can attribute and filter results.

CHUNKING STRATEGY (from phase1.txt):
- PG Essays:         400-600 words, split at paragraph/idea breaks
- YC Blog:           300-500 words, split at heading/section breaks
- Startup School:    500-800 words, split at topic shifts
- HN Threads:        1 chunk per post (already short)
- Companies:         1 chunk per company row (one sentence each)
- YC App Questions:  1 chunk per question/section

Usage:
  python src/chunker.py
"""

import csv
import json
import os
import re

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "chunks.json")

PG_DIR = os.path.join(DATA_DIR, "pg_essays")
YC_BLOG_DIR = os.path.join(DATA_DIR, "yc_blog")
STARTUP_SCHOOL_DIR = os.path.join(DATA_DIR, "startup_school")
COMPANIES_CSV = os.path.join(DATA_DIR, "companies.csv")
HN_FILE = os.path.join(DATA_DIR, "hn_threads.json")
YC_APP_FILE = os.path.join(DATA_DIR, "yc_application_questions.txt")


# ---------------------------------------------------------------------------
# TOPIC TAGGING
# ---------------------------------------------------------------------------
# These keyword → tag mappings let us auto-tag chunks.
# Each chunk gets 1-3 tags based on keyword matches in the text.
# This is a simple but effective approach — a real production system
# might use an LLM for tagging, but keyword matching works well for
# well-defined domains like YC/startups.

ALLOWED_TOPIC_TAGS = {
    "fundraising", "pmf", "hiring", "applying", "growth", "idea",
    "pricing", "investors", "team", "product", "market",
    "competition", "culture", "technical",
}


TOPIC_KEYWORDS: dict[str, list[str]] = {
    "fundraising": [
        "fundrais", "investor", "pitch", "valuation", "term sheet",
        "seed round", "series a", "venture capital", "vc ", "raise money",
        "cap table", "dilution", "SAFE", "convertible note", "angel",
    ],
    "pmf": [
        "product market fit", "product-market fit", "pmf",
        "retention", "churn", "engagement", "growth rate",
    ],
    "hiring": [
        "hiring", "recruit", "talent", "team building", "co-founder",
        "cofounder", "employee", "engineer", "culture fit",
    ],
    "applying": [
        "yc application", "apply to yc", "applying to yc", "yc interview",
        "application question", "yc batch", "demo day",
    ],
    "growth": [
        "growth", "scaling", "scale ", "traction", "metrics", "kpi",
        "revenue", "mrr", "arr", "conversion", "acquisition",
    ],
    "idea": [
        "startup idea", "idea validation", "problem worth solving",
        "pivot", "market size", "tam ", "opportunity",
    ],
    "pricing": [
        "pricing", "price", "priced", "freemium", "subscription",
        "plan", "plans", "monetization", "monetisation",
    ],
    "investors": [
        "investor", "investors", "vc", "venture capital", "angel",
        "term sheet", "cap table",
    ],
    "team": [
        "team", "cofounder", "co-founder", "founding team",
        "early team", "teammate", "manager",
    ],
    "product": [
        "product", "mvp", "prototype", "feature", "roadmap",
        "shipping", "build", "user feedback",
    ],
    "market": [
        "market", "customer", "segment", "demand", "go-to-market",
        "gtm", "distribution", "channel",
    ],
    "competition": [
        "competition", "competitor", "moat", "differentiation",
        "incumbent", "alternative",
    ],
    "culture": [
        "company culture", "values", "mission", "morale",
        "remote work", "team dynamic",
    ],
    "technical": [
        "programming", "software", "engineer", "code", "api",
        "database", "infrastructure", "deploy", "architecture",
    ],
}


def auto_tag(text: str) -> list[str]:
    """
    Automatically assign 1-3 topic tags based on keyword matches.

    HOW IT WORKS:
    - Scan the text for keywords from each topic category
    - Count how many keywords match for each topic
    - Return the top 1-3 topics with the most matches
    - If nothing matches, return []
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for tag, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[tag] = score

    if not scores:
        return []

    # Sort by score descending, take top 3
    sorted_tags = sorted(scores, key=scores.get, reverse=True)
    return [tag for tag in sorted_tags if tag in ALLOWED_TOPIC_TAGS][:3]


# ---------------------------------------------------------------------------
# QUALITY TIER ASSIGNMENT
# ---------------------------------------------------------------------------
# Quality tiers help the RAG system prioritize sources:
#   Tier 1 = Highest quality (PG essays, YC partner posts)
#   Tier 2 = High quality (Launch HN, companies, curated lectures)
#   Tier 3 = Community content (general HN threads)

def assign_quality_tier(source_type: str, author: str = "") -> int:
    """
    Assign a quality tier based on source and author.

    Tier 1: Paul Graham essays, YC partner blog posts
    Tier 2: Startup School lectures, Launch HN, company descriptions
    Tier 3: General HN community posts
    """
    if source_type == "pg_essay":
        return 1  # PG essays are always tier 1

    if source_type == "yc_blog":
        # Check if author is a known YC partner
        yc_partners = [
            "dalton caldwell", "michael seibel", "jared friedman",
            "gustaf alströmer", "gustaf alstromer", "garry tan",
            "kat manalac", "kevin hale", "adora cheung",
            "kirsty nathoo", "tim brady",
        ]
        if any(p in author.lower() for p in yc_partners):
            return 1
        return 2  # Other blog authors

    if source_type == "startup_school":
        return 1  # Curated lectures from YC partners

    if source_type == "company":
        return 2

    if source_type == "hn":
        return 3  # Community content

    if source_type == "yc_app_questions":
        return 1  # Official YC content

    return 3


# ---------------------------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------------------------
# Phase 0 already did heavy cleaning (HTML removal, filler words, etc.)
# Here we just do light normalization that's needed before chunking.

def clean_text(text: str) -> str:
    """
    Light text cleaning — DON'T redo what Phase 0 already did.

    What we DO here:
    - Normalize curly quotes → straight quotes (important for consistency)
    - Collapse excessive whitespace
    - Strip leading/trailing whitespace

    What we DON'T do (already done in Phase 0):
    - Remove HTML tags (already stripped)
    - Remove filler words from transcripts (already cleaned)
    - Filter by word count (blog posts already filtered to 300+ words)
    """
    # Normalize curly/smart quotes to straight quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # single quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # double quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")  # en/em dashes

    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces (but not newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# METADATA HEADER PARSING
# ---------------------------------------------------------------------------
# Phase 0 saved blog/lecture files with metadata headers like:
#   Title: How to Talk to Users
#   Speaker: Gustaf Alströmer
#   Source: https://youtube.com/watch?v=...
#   ---
# We parse these directly instead of re-extracting.

def parse_metadata_header(text: str) -> tuple[dict, str]:
    """
    Parse the metadata header from a Phase 0 file.

    Returns:
        (metadata_dict, remaining_body_text)

    If no header is found, returns empty dict and the full text.
    """
    metadata = {}

    # Check if file has a metadata header (lines of "Key: Value" followed by "---")
    if "---" not in text[:500]:
        return metadata, text

    lines = text.split("\n")
    body_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "---":
            body_start = i + 1
            break
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            metadata[key.strip().lower()] = value.strip()

    body = "\n".join(lines[body_start:]).strip()
    return metadata, body


def extract_author_from_metadata(metadata: dict, default: str = "") -> str:
    """Return author from metadata, falling back to speaker if needed."""
    author = metadata.get("author", "").strip()
    if author:
        return author
    speaker = metadata.get("speaker", "").strip()
    if speaker:
        return speaker
    return default


# ---------------------------------------------------------------------------
# CHUNKING FUNCTIONS
# ---------------------------------------------------------------------------
# Each source has its own chunking strategy because the content is different.
# This is the KEY insight of phase1.txt: "Don't chunk everything the same way."


def chunk_by_paragraphs(text: str, min_words: int, max_words: int) -> list[str]:
    """
    PARAGRAPH-BASED CHUNKING — used for PG essays and YC blog posts.

    HOW IT WORKS:
    1. Split text into paragraphs (separated by blank lines)
    2. Accumulate paragraphs into a chunk until we hit the word target
    3. When we exceed max_words, save the chunk and start a new one
    4. Always break at paragraph boundaries — never mid-sentence

    WHY paragraph boundaries?
    - A paragraph usually = one complete thought/argument
    - Breaking mid-paragraph creates chunks that start/end awkwardly
    - This preserves the author's logical flow

    Args:
        text: The full text to chunk
        min_words: Don't create chunks smaller than this (merge with next)
        max_words: Start a new chunk after exceeding this
    """
    # Split on double newlines (paragraph breaks)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If adding this paragraph would exceed max AND we already
        # have enough content, save current chunk first
        if current_words + para_words > max_words and current_words >= min_words:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_words = 0

        current_chunk.append(para)
        current_words += para_words

    # Don't forget the last chunk!
    if current_chunk:
        last_chunk = "\n\n".join(current_chunk)
        # If last chunk is too small, merge it with the previous one
        if current_words < min_words and chunks:
            chunks[-1] += "\n\n" + last_chunk
        else:
            chunks.append(last_chunk)

    return chunks


def split_by_sentence_limit(text: str, max_words: int) -> list[str]:
    """Fallback splitter: split long text by sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current).strip())
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


def split_oversized_chunk(
    text: str,
    target_max_words: int = 700,
    hard_max_words: int = 800,
) -> list[str]:
    """
    Split chunks that exceed hard_max_words.

    Preference order:
    1) paragraph boundary near target_max_words
    2) sentence boundary fallback when a single paragraph is too long
    """
    word_count = len(text.split())
    if word_count <= hard_max_words:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    split_chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if current_parts and current_words + para_words > target_max_words:
            candidate = "\n\n".join(current_parts).strip()
            if len(candidate.split()) > hard_max_words:
                split_chunks.extend(split_by_sentence_limit(candidate, hard_max_words))
            else:
                split_chunks.append(candidate)
            current_parts = []
            current_words = 0

        if para_words > hard_max_words:
            if current_parts:
                split_chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_words = 0
            split_chunks.extend(split_by_sentence_limit(para, hard_max_words))
            continue

        current_parts.append(para)
        current_words += para_words

    if current_parts:
        candidate = "\n\n".join(current_parts).strip()
        if len(candidate.split()) > hard_max_words:
            split_chunks.extend(split_by_sentence_limit(candidate, hard_max_words))
        else:
            split_chunks.append(candidate)

    return [chunk for chunk in split_chunks if chunk.strip()]


def enforce_chunk_limits(
    text_chunks: list[str],
    target_min_words: int = 150,
    hard_min_words: int = 100,
    hard_max_words: int = 800,
) -> list[str]:
    """Enforce chunk size rules, including splitting oversized chunks."""
    normalized: list[str] = []

    for chunk_text in text_chunks:
        for subchunk in split_oversized_chunk(
            chunk_text,
            target_max_words=700,
            hard_max_words=hard_max_words,
        ):
            subchunk = subchunk.strip()
            if not subchunk:
                continue
            if len(subchunk.split()) < hard_min_words:
                continue
            normalized.append(subchunk)

    if not normalized:
        return []

    merged: list[str] = []
    for chunk_text in normalized:
        if merged and len(chunk_text.split()) < target_min_words:
            candidate = merged[-1].strip() + "\n\n" + chunk_text
            if len(candidate.split()) <= hard_max_words:
                merged[-1] = candidate
                continue
        merged.append(chunk_text)

    return merged


def is_pg_relevant(title: str, text: str) -> bool:
    """Heuristic PG relevance check using title or first paragraph."""
    first_para = re.split(r"\n\s*\n", text.strip())[0] if text.strip() else ""
    probe = f"{title}\n{first_para}".lower()

    positive = [
        "startup", "startups", "founder", "founders", "investor",
        "investors", "building companies", "build companies", "company",
    ]
    negative = [
        "parenting", "kids", "children", "city", "cities",
        "personal life", "marriage", "relationship", "family",
    ]

    has_positive = any(keyword in probe for keyword in positive)
    has_negative = any(keyword in probe for keyword in negative)

    if not has_positive:
        return False
    if has_negative and not any(k in probe for k in ("startup", "founder", "investor")):
        return False
    return True


def chunk_transcript(text: str, min_words: int = 500, max_words: int = 800) -> list[str]:
    """
    TRANSCRIPT CHUNKING — for Startup School lectures.

    Transcripts are tricky because they're one long stream of text without
    clear paragraph breaks. We use sentence-based chunking instead:

    1. Split into sentences
    2. Accumulate until we hit the word target
    3. Try to break at natural topic transitions (signaled by phrases like
       "so let's talk about", "next", "the second thing", etc.)

    WHY larger chunks for transcripts?
    - Spoken language is less dense than written text
    - A single "um, so basically what I mean is..." takes 10 words to
      convey what an essay says in 3 words
    - Transcripts need more words to capture a complete idea
    """
    # Split into sentences (rough but works for transcripts)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Topic transition signals — good places to break
    transition_signals = [
        "so let's", "let's talk", "next", "the second",
        "the third", "another", "moving on", "now I want",
        "let me", "all right so", "so to summarize",
        "here's what", "to sum up", "finally",
    ]

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_words = 0

    for sentence in sentences:
        s_words = len(sentence.split())

        # Check for topic transition
        is_transition = any(
            signal in sentence.lower() for signal in transition_signals
        )

        # Break at transitions if we have enough content
        if is_transition and current_words >= min_words:
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_words = 0

        # Break if exceeding max words
        if current_words + s_words > max_words and current_words >= min_words:
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_words = 0

        current_sentences.append(sentence)
        current_words += s_words

    if current_sentences:
        last = " ".join(current_sentences)
        if current_words < min_words and chunks:
            chunks[-1] += " " + last
        else:
            chunks.append(last)

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE-SPECIFIC PROCESSORS
# ═══════════════════════════════════════════════════════════════════════════
# Each processor reads raw files, chunks them, and returns a list of
# chunk dicts ready for chunks.json.

def process_pg_essays() -> list[dict]:
    """
    Process Paul Graham essays → chunks.

    Strategy: 400-600 words per chunk, split at paragraph breaks.
    PG essays don't have metadata headers — the filename IS the title.
    Author is always "Paul Graham".
    """
    print("\n📄 Processing PG Essays...")
    chunks = []

    if not os.path.exists(PG_DIR):
        print("  ⚠ PG essays directory not found. Run scraper first.")
        return []

    files = sorted(f for f in os.listdir(PG_DIR) if f.endswith(".txt"))
    chunk_counter = 0

    for filename in files:
        filepath = os.path.join(PG_DIR, filename)
        title = filename.replace(".txt", "")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        text = clean_text(text)

        # Remove "Thanks to ... for reading drafts" at the end
        text = re.sub(
            r"\n\s*Thanks\s+to\s+.*?for reading drafts.*$",
            "", text, flags=re.DOTALL | re.IGNORECASE
        )

        if not is_pg_relevant(title, text):
            continue

        # Chunk by paragraphs: target 150-700 words
        text_chunks = chunk_by_paragraphs(text, min_words=150, max_words=700)
        text_chunks = enforce_chunk_limits(text_chunks)

        for chunk_text in text_chunks:
            word_count = len(chunk_text.split())
            topic_tags = auto_tag(chunk_text)
            if not topic_tags:
                continue

            chunk_counter += 1
            chunks.append({
                "chunk_id": f"pg_{chunk_counter:04d}",
                "text": chunk_text,
                "source_type": "pg_essay",
                "title": title,
                "author": "Paul Graham",
                "date": "",
                "topic_tags": topic_tags,
                "quality_tier": 1,
                "word_count": word_count,
            })

    print(f"  ✅ {len(files)} essays → {len(chunks)} chunks")
    return chunks


def process_yc_blog() -> list[dict]:
    """
    Process YC Blog posts → chunks.

    Strategy: 300-500 words per chunk, split at paragraph breaks.
    Blog files have metadata headers (Title, Author, Date, URL).
    """
    print("\n📄 Processing YC Blog Posts...")
    chunks = []

    if not os.path.exists(YC_BLOG_DIR):
        print("  ⚠ YC blog directory not found. Run scraper first.")
        return []

    files = sorted(f for f in os.listdir(YC_BLOG_DIR) if f.endswith(".txt"))
    chunk_counter = 0

    for filename in files:
        filepath = os.path.join(YC_BLOG_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Parse the metadata header that Phase 0 already created
        metadata, body = parse_metadata_header(text)
        title = metadata.get("title", filename.replace(".txt", ""))
        author = extract_author_from_metadata(metadata)
        date = metadata.get("date", "")

        body = clean_text(body)

        # Chunk by paragraphs: target 150-700 words
        text_chunks = chunk_by_paragraphs(body, min_words=150, max_words=700)
        text_chunks = enforce_chunk_limits(text_chunks)

        for chunk_text in text_chunks:
            word_count = len(chunk_text.split())
            topic_tags = auto_tag(chunk_text)
            if not topic_tags:
                continue

            chunk_counter += 1
            chunks.append({
                "chunk_id": f"yc_blog_{chunk_counter:04d}",
                "text": chunk_text,
                "source_type": "yc_blog",
                "title": title,
                "author": author,
                "date": date,
                "topic_tags": topic_tags,
                "quality_tier": assign_quality_tier("yc_blog", author),
                "word_count": word_count,
            })

    print(f"  ✅ {len(files)} posts → {len(chunks)} chunks")
    return chunks


def process_startup_school() -> list[dict]:
    """
    Process Startup School transcripts → chunks.

    Strategy: 500-800 words per chunk, split at topic transitions.
    Transcript files have metadata headers (Title, Speaker, Source).
    """
    print("\n📄 Processing Startup School Transcripts...")
    chunks = []

    if not os.path.exists(STARTUP_SCHOOL_DIR):
        print("  ⚠ Startup School directory not found. Run scraper first.")
        return []

    files = sorted(f for f in os.listdir(STARTUP_SCHOOL_DIR) if f.endswith(".txt"))
    chunk_counter = 0

    for filename in files:
        filepath = os.path.join(STARTUP_SCHOOL_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Parse metadata header
        metadata, body = parse_metadata_header(text)
        title = metadata.get("title", filename.replace(".txt", ""))
        speaker = extract_author_from_metadata(metadata)
        date = metadata.get("date", "")

        body = clean_text(body)

        # Chunk transcripts: target 150-700 words with topic detection
        text_chunks = chunk_transcript(body, min_words=150, max_words=700)
        text_chunks = enforce_chunk_limits(text_chunks)

        for chunk_text in text_chunks:
            word_count = len(chunk_text.split())
            topic_tags = auto_tag(chunk_text)
            if not topic_tags:
                continue

            chunk_counter += 1
            chunks.append({
                "chunk_id": f"ss_{chunk_counter:04d}",
                "text": chunk_text,
                "source_type": "startup_school",
                "title": title,
                "author": speaker,
                "date": date,
                "topic_tags": topic_tags,
                "quality_tier": 1,
                "word_count": word_count,
            })

    print(f"  ✅ {len(files)} lectures → {len(chunks)} chunks")
    return chunks


def process_hn_threads() -> list[dict]:
    """
    Process Hacker News threads → chunks.

    Strategy: Each post/thread is ONE chunk (they're already short).
    No splitting needed — just tag and filter.
    """
    print("\n📄 Processing HN Threads...")
    chunks = []

    if not os.path.exists(HN_FILE):
        print("  ⚠ HN threads file not found. Run scraper first.")
        return []

    with open(HN_FILE, "r", encoding="utf-8") as f:
        threads = json.load(f)

    chunk_counter = 0

    hn_title_keywords = [
        "yc", "y combinator", "startup", "founder",
        "apply yc", "launch hn", "got into yc",
    ]

    for thread in threads:
        title = thread.get("title", "")
        story_text = thread.get("story_text", "")
        author = thread.get("author", "")
        points = thread.get("points", 0)
        try:
            points = int(points)
        except (TypeError, ValueError):
            points = 0
        date = thread.get("created_at", "")

        title_lower = title.lower()
        if not any(keyword in title_lower for keyword in hn_title_keywords):
            continue

        if points < 15:
            continue

        # Build the chunk text from title + story text
        text_parts = [title]
        if story_text:
            text_parts.append(clean_text(story_text))
        text = "\n\n".join(text_parts)

        word_count = len(text.split())

        # HN minimum length
        if word_count < 200:
            continue

        # Skip pure announcements
        announcement_keywords = [
            "applications are open", "deadline", "is now accepting",
            "office hours", "apply now",
        ]
        if any(kw in text.lower() for kw in announcement_keywords):
            if word_count < 50:  # Only skip if they're short announcements
                continue

        chunk_counter += 1

        # Launch HN posts get tier 2, regular posts get tier 3
        is_launch = "launch hn" in title.lower()
        tier = 2 if is_launch else 3

        topic_tags = auto_tag(text)
        if not topic_tags:
            continue

        chunks.append({
            "chunk_id": f"hn_{chunk_counter:04d}",
            "text": text,
            "source_type": "hn",
            "title": title,
            "author": author,
            "date": date,
            "topic_tags": topic_tags,
            "quality_tier": tier,
            "word_count": word_count,
            "points": points,  # extra metadata for HN
        })

    print(f"  ✅ {len(threads)} threads → {len(chunks)} chunks")
    return chunks


def process_companies() -> list[dict]:
    """
    Process YC Company Directory → chunks.

    Strategy: Each company = one chunk, formatted as a descriptive sentence.
    Example: "Stripe is an Active Fintech company from the USA, batch W09,
             that builds payment infrastructure for the internet."
    """
    print("\n📄 Processing YC Companies...")
    chunks = []

    if not os.path.exists(COMPANIES_CSV):
        print("  ⚠ Companies CSV not found. Run scraper first.")
        return []

    with open(COMPANIES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    chunk_counter = 0

    for row in rows:
        name = row.get("name", "").strip()
        batch = row.get("batch", "").strip()
        industry = row.get("industry", "").strip()
        region = row.get("region", "").strip()
        one_liner = row.get("one_liner", "").strip()
        status = row.get("status", "Active").strip()
        website = row.get("website", "").strip()
        team_size = row.get("team_size", "").strip()

        if not name or not one_liner:
            continue

        status = status or "Unknown"
        industry = industry or "Unknown"
        region = region or "Unknown"
        batch = batch or "Unknown"
        description = one_liner.rstrip(".") or "products and services"
        team_size = team_size or "Unknown"
        website = website or "Unknown"

        text = (
            f"{name} is a YC startup from batch {batch} and is currently marked as {status}. "
            f"The company operates in {industry} and is associated with the region {region}. "
            f"Its one-line company description is: {description}. "
            f"From a startup profile perspective, this suggests the product focus, target market, and execution model used by the founding team. "
            f"The reported team size is {team_size}, which gives directional context on scale and operating maturity. "
            f"The public website listed for the company is {website}. "
            f"This company profile can be used for YC-style comparisons involving startup ideas, market positioning, growth stage, and founder strategy across companies in similar categories."
        )

        topic_tags = auto_tag(text + " " + industry)
        if not topic_tags:
            continue

        chunk_counter += 1
        chunks.append({
            "chunk_id": f"co_{chunk_counter:04d}",
            "text": text,
            "source_type": "company",
            "title": name,
            "author": "",
            "date": "",
            "topic_tags": topic_tags,
            "quality_tier": 2,
            "word_count": len(text.split()),
        })

    print(f"  ✅ {len(rows)} companies → {len(chunks)} chunks")
    return chunks


def process_yc_application_questions() -> list[dict]:
    """
    Process YC Application Questions → chunks.

    Strategy: Split by section (COMPANY, FOUNDERS, PROGRESS, etc.)
    Each section becomes one chunk with all its questions.
    """
    print("\n📄 Processing YC Application Questions...")
    chunks = []

    if not os.path.exists(YC_APP_FILE):
        print("  ⚠ YC application questions file not found. Run scraper first.")
        return []

    with open(YC_APP_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_text(text)

    # Split by section headers (lines of all caps followed by dashes)
    sections = re.split(r"\n([A-Z]+)\n-+\n", text)

    # First element is the intro, then alternating section_name / section_body
    chunk_counter = 0

    # Process intro as its own chunk
    intro = sections[0].strip()
    if intro and len(intro.split()) >= 20:
        chunk_counter += 1
        chunks.append({
            "chunk_id": f"yc_app_{chunk_counter:04d}",
            "text": intro,
            "source_type": "yc_app_questions",
            "title": "YC Application Questions - Overview",
            "author": "Y Combinator",
            "date": "",
            "topic_tags": ["applying"],
            "quality_tier": 1,
            "word_count": len(intro.split()),
        })

    # Process each section
    for i in range(1, len(sections) - 1, 2):
        section_name = sections[i].strip()
        section_body = sections[i + 1].strip() if i + 1 < len(sections) else ""

        if not section_body:
            continue

        full_text = f"YC Application - {section_name} Section:\n\n{section_body}"
        word_count = len(full_text.split())

        if word_count < 10:
            continue

        chunk_counter += 1
        chunks.append({
            "chunk_id": f"yc_app_{chunk_counter:04d}",
            "text": full_text,
            "source_type": "yc_app_questions",
            "title": f"YC Application Questions - {section_name}",
            "author": "Y Combinator",
            "date": "",
            "topic_tags": ["applying"],
            "quality_tier": 1,
            "word_count": word_count,
        })

    print(f"  ✅ {len(chunks)} chunks from YC application questions")
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """
    Remove near-duplicate chunks.

    HOW IT WORKS:
    - Normalize each chunk's text (lowercase, collapse spaces)
    - Compare the first 200 chars as a fingerprint
    - If two chunks have the same fingerprint, keep the longer one

    WHY 200 chars?
    - Full text comparison is expensive for 7000 chunks
    - The first 200 chars are usually unique enough
    - This catches exact duplicates and near-duplicates where only
      the ending differs
    """
    seen: dict[str, int] = {}  # fingerprint → index in unique list
    unique: list[dict] = []

    for chunk in chunks:
        # Create fingerprint: first 200 chars, normalized
        normalized = re.sub(r"\s+", " ", chunk["text"].lower().strip())
        fingerprint = normalized[:200]

        if fingerprint in seen:
            # Keep the longer version
            existing_idx = seen[fingerprint]
            if chunk["word_count"] > unique[existing_idx]["word_count"]:
                unique[existing_idx] = chunk
        else:
            seen[fingerprint] = len(unique)
            unique.append(chunk)

    removed = len(chunks) - len(unique)
    if removed > 0:
        print(f"  🔄 Removed {removed} near-duplicate chunks")
    return unique


def is_pure_announcement(text: str) -> bool:
    """Heuristic filter for short announcement-style chunks."""
    text_lower = text.lower()
    patterns = [
        "applications are open",
        "is now accepting applications",
        "deadline",
        "apply now",
        "registration is open",
    ]
    return any(pattern in text_lower for pattern in patterns)


def is_startup_related(text: str) -> bool:
    """Heuristic filter for startup/YC relevance."""
    text_lower = text.lower()
    keywords = [
        "startup", "founder", "company", "yc", "y combinator", "investor",
        "product", "users", "customer", "market", "growth", "hiring",
        "fundrais", "application", "launch", "mvp", "team", "revenue",
    ]
    return any(keyword in text_lower for keyword in keywords)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_chunker():
    """Main entry point: process all sources and save chunks.json."""
    print("=" * 60)
    print("  Phase 1 — Data Cleaning & Chunking")
    print("=" * 60)

    all_chunks: list[dict] = []

    # Process each source
    all_chunks.extend(process_pg_essays())
    all_chunks.extend(process_yc_blog())
    all_chunks.extend(process_startup_school())
    all_chunks.extend(process_hn_threads())
    all_chunks.extend(process_companies())

    # Hard filter: remove chunks below 100 words
    before = len(all_chunks)
    all_chunks = [
        c for c in all_chunks
        if 100 <= c["word_count"] <= 800
    ]
    filtered = before - len(all_chunks)
    if filtered > 0:
        print(f"\n🗑 Removed {filtered} short/low-value chunks")

    before_tags = len(all_chunks)
    all_chunks = [c for c in all_chunks if c.get("topic_tags")]
    removed_no_tags = before_tags - len(all_chunks)
    if removed_no_tags > 0:
        print(f"  🏷 Removed {removed_no_tags} chunks with no allowed tags")

    before_announcement = len(all_chunks)
    all_chunks = [
        c for c in all_chunks
        if c["source_type"] not in ("hn", "yc_blog") or not is_pure_announcement(c["text"])
    ]
    removed_announcement = before_announcement - len(all_chunks)
    if removed_announcement > 0:
        print(f"  🗞 Removed {removed_announcement} pure announcements")

    before_topic = len(all_chunks)
    all_chunks = [
        c for c in all_chunks
        if c["source_type"] == "company" or is_startup_related(c["text"])
    ]
    removed_offtopic = before_topic - len(all_chunks)
    if removed_offtopic > 0:
        print(f"  🎯 Removed {removed_offtopic} off-topic chunks")

    # Deduplicate
    print("\n🔍 Deduplicating...")
    all_chunks = deduplicate_chunks(all_chunks)

    # Save to chunks.json
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  ✅ Phase 1 Complete!")
    print(f"{'=' * 60}")
    print(f"\n📊 Chunk Summary:")

    # Count by source
    by_source: dict[str, int] = {}
    by_tier: dict[int, int] = {}
    total_words = 0

    for chunk in all_chunks:
        src = chunk["source_type"]
        tier = chunk["quality_tier"]
        by_source[src] = by_source.get(src, 0) + 1
        by_tier[tier] = by_tier.get(tier, 0) + 1
        total_words += chunk["word_count"]

    print(f"\n  {'Source':<25} {'Chunks':>8}")
    print(f"  {'-'*25} {'-'*8}")
    for src, count in sorted(by_source.items()):
        print(f"  {src:<25} {count:>8}")
    print(f"  {'-'*25} {'-'*8}")
    print(f"  {'TOTAL':<25} {len(all_chunks):>8}")

    print(f"\n  Quality Tiers:")
    for tier in sorted(by_tier):
        print(f"    Tier {tier}: {by_tier[tier]} chunks")

    print(f"\n  Total words: {total_words:,}")
    print(f"  Avg words/chunk: {total_words // max(len(all_chunks), 1)}")
    print(f"\n  📁 Output: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    run_chunker()

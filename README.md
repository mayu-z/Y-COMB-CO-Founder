# 🚀 YC Co-Founder

An AI-powered startup advisor built on real Y Combinator knowledge — Paul Graham essays, YC partner blog posts, Startup School lectures, and data from 1,494 YC-backed companies.

Built as a full RAG (Retrieval-Augmented Generation) pipeline from scratch.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-green)
![Groq](https://img.shields.io/badge/LLM-Llama%203.3%2070B-orange)

---

## What It Does

### 💬 Ask YC
Ask any question about startups, fundraising, YC applications, or founder strategy. Get answers grounded in real YC knowledge with source attribution.

> **Q:** "How do I get into YC?"  
> **A:** According to YC's Startup School, "It's never too early to apply" and "the best thing that gives you a shot is just hitting the apply button." They fund people who haven't quit their jobs yet, are still in school, or are just thinking about an idea…  
> *Sources: [pg_essay] A Fundraising Survival Guide — Paul Graham, [startup_school] YC Channel Talk 08*

### 🔍 Evaluate My Startup
Describe your startup and get a YC partner-style assessment:
- What's genuinely interesting about your idea
- What a YC partner would push back on
- Similar YC companies that got funded (with batch and status)
- One question a YC interviewer would ask you
- Honest fit assessment

### 🏢 Browse YC Companies
Filter and search through 1,494 YC-backed companies by industry, batch, status, or name.

---

## Knowledge Base

| Source | Count | Description |
|---|---|---|
| Paul Graham Essays | 229 essays → 325 chunks | Scraped from paulgraham.com |
| YC Blog Posts | 124 posts → 321 chunks | Partner posts from ycombinator.com/blog |
| Startup School | 20 lectures → 123 chunks | YouTube transcript extraction |
| Hacker News | ~500 threads → 541 chunks | YC-related threads via Algolia API |
| YC Companies | 1,494 companies → 1,494 chunks | Company directory with batch, industry, status |
| **Total** | **2,804 chunks** | Embedded in ChromaDB with `all-mpnet-base-v2` |

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  Raw Data   │────▶│  Chunker    │────▶│  chunks.json │
│  (376 files)│     │  (Phase 1)  │     │  (2,804)     │
└─────────────┘     └─────────────┘     └──────┬───────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │    Embedder       │
                                    │    (Phase 2)      │
                                    │  all-mpnet-base-v2│
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │    ChromaDB       │
                                    │  yc_knowledge     │
                                    │  (cosine, 768d)   │
                                    └────────┬─────────┘
                                             │
                              ┌──────────────┼──────────────┐
                              ▼              ▼              ▼
                     ┌──────────────┐ ┌────────────┐ ┌────────────┐
                     │  Retriever   │ │  RAG Engine │ │  Evaluator │
                     │  (hybrid)    │ │  (Groq LLM) │ │  (Phase 4) │
                     └──────────────┘ └──────┬─────┘ └──────┬─────┘
                                             │              │
                                             ▼              ▼
                                    ┌──────────────────────────┐
                                    │     Streamlit App        │
                                    │  Ask YC │ Evaluate │ Browse│
                                    └──────────────────────────┘
```

**Retrieval pipeline:** Hybrid search (semantic via ChromaDB + keyword matching) → deduplicate by chunk ID and title → re-rank by quality tier → diversity filter (max one result per source title) → top 5 results → LLM generates attributed answer.

---

## Project Structure

```
yc-cofounder/
├── app.py                    ← Streamlit UI (3 tabs)
├── .env                      ← GROQ_API_KEY (not committed)
├── requirements.txt
│
├── src/
│   ├── scraper.py            ← Phase 0: Data collection (6 sources)
│   ├── process_yc_blog.py    ← YC blog CSV processor
│   ├── chunker.py            ← Phase 1: Cleaning, chunking, tagging
│   ├── validate_chunks.py    ← Chunk quality validator
│   ├── embedder.py           ← Phase 2: Embed chunks → ChromaDB
│   ├── retriever.py          ← Phase 2: Hybrid retrieval engine
│   ├── rag.py                ← Phase 3: Groq LLM integration
│   └── evaluator.py          ← Phase 4: Startup evaluation feature
│
├── data/
│   ├── raw/
│   │   ├── pg_essays/        ← 229 essay .txt files
│   │   ├── yc_blog/          ← 124 blog post .txt files
│   │   ├── startup_school/   ← 20 lecture transcript .txt files
│   │   ├── companies.csv     ← 1,494 YC companies
│   │   ├── hn_threads.json   ← Hacker News YC threads
│   │   └── yc_application_questions.txt
│   ├── processed/
│   │   └── chunks.json       ← 2,804 tagged chunks
│   └── vectorstore/
│       └── chroma.sqlite3    ← ChromaDB persistent store
│
└── phase0-4.txt              ← Build roadmap docs
```

---

## Setup

### Prerequisites
- Python 3.12+
- A [Groq API key](https://console.groq.com) (free tier available)

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/yc-cofounder.git
cd yc-cofounder

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install all dependencies
pip install requests beautifulsoup4 lxml youtube-transcript-api \
            sentence-transformers chromadb groq python-dotenv streamlit
```

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Rebuild From Scratch

If you want to rebuild the full pipeline from raw data:

```bash
# Phase 0 — Scrape all sources
python src/scraper.py all

# Phase 1 — Clean and chunk
python src/chunker.py

# Phase 1 — Validate chunks
python src/validate_chunks.py

# Phase 2 — Embed into ChromaDB
python src/embedder.py

# Phase 2 — Test retrieval
python src/retriever.py

# Phase 3 — Test RAG engine
python src/rag.py --test

# Phase 4 — Test evaluator
python src/evaluator.py --test

# Phase 5 — Launch app
streamlit run app.py
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.12 |
| **LLM** | Llama 3.3 70B via [Groq](https://groq.com) (fast, free tier) |
| **Embeddings** | `all-mpnet-base-v2` (sentence-transformers, runs locally) |
| **Vector Store** | ChromaDB (persistent, local, cosine similarity) |
| **Frontend** | Streamlit |
| **Scraping** | requests, BeautifulSoup, youtube-transcript-api |

---

## Key Design Decisions

- **Hybrid retrieval** — Semantic search alone misses keyword-heavy queries. Combining semantic + keyword search with deduplication and quality-tier re-ranking gives better results.
- **Quality tiers** — PG essays and YC partner posts (tier 1) are ranked above HN threads and company descriptions (tier 2-3) for advice queries.
- **Diversity filter** — No single essay/post title appears more than once in results, preventing one long essay from dominating answers.
- **Source attribution** — Every answer cites where the information comes from. The LLM is instructed to never invent quotes, company names, or statistics.
- **Groq over OpenAI/Anthropic** — Free tier with fast inference on Llama 3.3 70B. Good enough quality for a RAG pipeline where the context does the heavy lifting.
- **Local embeddings** — `all-mpnet-base-v2` runs on CPU with no API costs. 768-dimensional vectors with strong semantic similarity performance.

---

## Build Phases

| Phase | What | Files |
|---|---|---|
| **0** | Data collection — scrape 6 sources into raw files | `scraper.py`, `process_yc_blog.py` |
| **1** | Cleaning & chunking — turn raw text into 2,804 tagged chunks | `chunker.py`, `validate_chunks.py` |
| **2** | Embeddings & retrieval — ChromaDB + hybrid search | `embedder.py`, `retriever.py` |
| **3** | RAG engine — connect retriever to Groq LLM | `rag.py` |
| **4** | Startup evaluator — YC-style assessment feature | `evaluator.py` |
| **5** | Streamlit app — 3-tab UI | `app.py` |

---

## License

This project is for educational purposes. YC company data and Paul Graham essays are publicly available. No proprietary YC content is included.

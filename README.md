# YC-Co-Founder

YC Co-Founder is a Retrieval-Augmented Generation (RAG) startup advisor built on Y Combinator knowledge.

It combines:
- Paul Graham essays
- YC blog posts
- Startup School transcripts
- YC company data
- YC application question context

The app provides grounded answers with sources and includes a benchmark pipeline for relevance, sourcing, and hallucination checks.

## Current Status

- LLM provider: Kimi K2 (OpenAI-compatible API)
- Retrieval: Hybrid (semantic + keyword) with re-ranking and diversity filtering
- Vector store: ChromaDB
- Frontend: Streamlit
- Benchmark set: 100 questions

## Main Features

### Ask YC
- Ask startup and YC questions
- Returns source-grounded answers
- Includes out-of-scope fallback behavior

### Evaluate My Startup
- YC-style startup assessment
- Similar company retrieval
- Structured founder feedback

### Browse YC Companies
- Explore indexed YC company data
- Filter and inspect metadata-rich entries

### Benchmark
- Run quality evaluation from the app or CLI
- Tracks relevance, source score, hallucination score, latency
- Saves JSON report to benchmark_results.json

## Project Structure

```text
yc-cofounder/
├── app.py
├── benchmark_questions.json
├── benchmark_results.json
├── requirements.txt
├── .env
├── src/
│   ├── scraper.py
│   ├── process_yc_blog.py
│   ├── chunker.py
│   ├── validate_chunks.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── rag.py
│   ├── evaluator.py
│   ├── benchmark.py
│   └── ragas_eval.py
└── data/
    ├── raw/
    ├── processed/chunks.json
    └── vectorstore/chroma.sqlite3
```

## Setup

### 1. Prerequisites

- Python 3.12+
- Kimi K2 API key

### 2. Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Mac/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create .env at project root:

```env
KIMI_K2_API_KEY=your_api_key_here
KIMI_K2_MODEL=moonshotai/kimi-k2-instruct
KIMI_K2_BASE_URL=https://integrate.api.nvidia.com/v1
```

Optional telemetry variables:

```env
OTEL_SERVICE_NAME=yc-benchmark
# Optional if you run a collector:
# OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces
```

## Run the App

Use the project root file app.py:

```bash
streamlit run app.py
```

Important: do not run streamlit run src/app.py (that file does not exist).

## CLI Benchmarking

### Run full benchmark (100 questions)

```bash
python src/benchmark.py --max-questions 100 --no-progress
```

### Run with explicit venv interpreter (recommended on Windows)

```powershell
$py = "c:/Users/Sakshi Singh/Downloads/q-paper-rag-/.venv/Scripts/python.exe"
$env:OTEL_SDK_DISABLED = 'true'
& $py "src/benchmark.py" --max-questions 100 --no-progress
```

### Output

- benchmark_results.json (full report)

## OpenTelemetry Notes

- Benchmark tracing is enabled in src/benchmark.py.
- If no OTLP endpoint is configured, spans may print to console.
- To suppress telemetry emission during local runs:

```powershell
$env:OTEL_SDK_DISABLED = 'true'
```

## Known Good Benchmark Snapshot

Latest 100-question run (project state at time of writing):

- Total questions: 100
- Avg relevance: 0.63
- Avg source score: 0.99
- Avg hallucination: 1.00
- Overall RAG score: 0.87

## Rebuild Pipeline from Raw Data

```bash
python src/scraper.py all
python src/chunker.py
python src/validate_chunks.py
python src/embedder.py
python src/retriever.py
python src/rag.py --test
python src/evaluator.py --test
streamlit run app.py
```

## Troubleshooting

### ModuleNotFoundError: No module named openai

Cause:
- You are using system Python instead of .venv Python.

Fix:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/benchmark.py --max-questions 100 --no-progress
```

### Error code 410: model reached end of life

Cause:
- The configured `KIMI_K2_MODEL` no longer exists on the provider side.

Fix:

```env
KIMI_K2_MODEL=<current-openai-compatible-model-name>
```

If you only want the UI to open for browsing and local inspection, you can still launch Streamlit, but any question or benchmark run that needs the model will fail until the model name is updated.

### streamlit run src/app.py fails

Cause:
- Wrong path.

Fix:

```bash
streamlit run app.py
```

### Benchmark runs fewer questions than requested

Cause:
- benchmark_questions.json has fewer entries than requested max.

Fix:
- Increase benchmark_questions.json entries or lower --max-questions.

## Tech Stack

- Python 3.12
- Kimi K2 (OpenAI-compatible API from nvidia)
- sentence-transformers (all-mpnet-base-v2)
- ChromaDB
- Streamlit
- OpenTelemetry (benchmark tracing)

## License

Educational project using public YC-related data sources.

clone it and try it.

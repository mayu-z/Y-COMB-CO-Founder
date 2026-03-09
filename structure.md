yc-cofounder/
├── data/
│   ├── raw/
│   │   ├── pg_essays/
│   │   ├── yc_blog/
│   │   ├── startup_school/
│   │   ├── hn_threads.json
│   │   ├── companies.csv
│   │   └── yc_application_questions.txt
│   └── processed/
│       └── chunks.json
├── src/
│   ├── scraper.py          ← Phase 0
│   ├── chunker.py          ← Phase 1
│   ├── embedder.py         ← Phase 2
│   ├── retriever.py        ← Phase 2
│   ├── rag.py              ← Phase 3
│   └── evaluator.py        ← Phase 4
├── app.py                  ← Phase 5
├── .env                    ← API key lives here, never commit this
└── requirements.txt
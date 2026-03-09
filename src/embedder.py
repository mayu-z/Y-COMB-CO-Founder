"""
Phase 2 — Embedder
Converts all chunks from chunks.json into embeddings
and stores them in a persistent ChromaDB collection.

Usage:
    python src/embedder.py
"""

import json
import time
import os
import chromadb
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "processed", "chunks.json")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore")
COLLECTION_NAME = "yc_knowledge"
MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 100


def load_chunks():
    """Load chunks from chunks.json."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from chunks.json")
    return chunks


def setup_chromadb():
    """Create / connect to persistent ChromaDB client and collection."""
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"ChromaDB collection '{COLLECTION_NAME}' ready  "
          f"(existing docs: {collection.count()})")
    return client, collection


def get_existing_ids(collection):
    """Return set of chunk IDs already in the collection."""
    existing = collection.get(include=[])          # id-only fetch
    return set(existing["ids"]) if existing["ids"] else set()


def build_metadata(chunk):
    """Convert a chunk dict into ChromaDB-safe metadata."""
    tags = chunk.get("topic_tags", [])
    if isinstance(tags, list):
        tags = ", ".join(tags)

    return {
        "source_type": chunk.get("source_type", ""),
        "title": chunk.get("title", ""),
        "author": chunk.get("author", ""),
        "topic_tags": tags,
        "quality_tier": int(chunk.get("quality_tier", 2)),
        "word_count": int(chunk.get("word_count", 0)),
    }


def embed_and_store(chunks, collection, model):
    """Embed new chunks in batches and upsert into ChromaDB."""
    existing_ids = get_existing_ids(collection)
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print("All chunks already embedded — nothing to do.")
        return 0

    skipped = len(chunks) - len(new_chunks)
    if skipped:
        print(f"Skipping {skipped} already-embedded chunks")

    total = len(new_chunks)
    print(f"Embedding {total} new chunks …")

    start = time.time()
    embedded_count = 0

    for i in range(0, total, BATCH_SIZE):
        batch = new_chunks[i : i + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [build_metadata(c) for c in batch]

        # Encode batch
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # Upsert into ChromaDB
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        embedded_count += len(batch)
        elapsed = time.time() - start
        print(f"  Embedded {embedded_count}/{total}  "
              f"({elapsed:.1f}s elapsed)")

    return embedded_count


def main():
    print("=" * 50)
    print("YC Co-Founder — Embedder (Phase 2)")
    print("=" * 50)

    # 1. Load
    chunks = load_chunks()

    # 2. ChromaDB
    client, collection = setup_chromadb()

    # 3. Model
    print(f"Loading embedding model: {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")

    # 4. Embed
    t0 = time.time()
    embedded = embed_and_store(chunks, collection, model)
    elapsed = time.time() - t0

    # 5. Summary
    print()
    print("─" * 40)
    print(f"Total chunks embedded this run : {embedded}")
    print(f"Time taken                     : {elapsed:.1f}s")
    print(f"Collection size                : {collection.count()}")
    print("─" * 40)
    print("Done ✓")


if __name__ == "__main__":
    main()

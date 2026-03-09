"""Phase 1 chunks.json validator.

Usage:
  python src/validate_chunks.py
  python src/validate_chunks.py --file data/processed/chunks.json

Checks:
- Required fields exist on every chunk
- topic_tags is a non-empty list with <= 3 tags
- word_count matches text word count (tolerates small drift)
- quality constraints from phase1.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_FIELDS = {
    "chunk_id",
    "text",
    "source_type",
    "title",
    "author",
    "date",
    "topic_tags",
    "quality_tier",
    "word_count",
}

ALLOWED_SOURCE_TYPES = {"pg_essay", "yc_blog", "startup_school", "hn", "company"}
ALLOWED_TIERS = {1, 2, 3}
MIN_WORDS_BY_SOURCE = {
    "pg_essay": 100,
    "yc_blog": 100,
    "startup_school": 100,
    "hn": 20,
    "company": 15,
}


def count_words(text: str) -> int:
    return len(text.split())


def is_pure_announcement(text: str) -> bool:
    text_lower = text.lower()
    markers = [
        "applications are open",
        "is now accepting applications",
        "apply now",
        "registration is open",
    ]
    return any(marker in text_lower for marker in markers)


def is_startup_related(text: str) -> bool:
    text_lower = text.lower()
    keywords = [
        "startup",
        "founder",
        "company",
        "yc",
        "y combinator",
        "investor",
        "product",
        "users",
        "customer",
        "market",
        "growth",
        "hiring",
        "fundrais",
        "application",
        "launch",
        "mvp",
        "team",
        "revenue",
    ]
    return any(keyword in text_lower for keyword in keywords)


def validate(file_path: Path) -> int:
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return 2

    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("ERROR: chunks file must be a JSON list")
        return 2

    errors: list[str] = []
    source_counts: Counter[str] = Counter()
    tier_counts: Counter[int] = Counter()
    seen_ids: set[str] = set()

    for index, chunk in enumerate(data):
        where = f"chunk[{index}]"
        if not isinstance(chunk, dict):
            errors.append(f"{where}: must be an object")
            continue

        missing = REQUIRED_FIELDS - set(chunk.keys())
        if missing:
            errors.append(f"{where}: missing fields {sorted(missing)}")
            continue

        chunk_id = chunk["chunk_id"]
        if chunk_id in seen_ids:
            errors.append(f"{where}: duplicate chunk_id '{chunk_id}'")
        seen_ids.add(chunk_id)

        source_type = chunk["source_type"]
        if source_type not in ALLOWED_SOURCE_TYPES:
            errors.append(f"{where}: invalid source_type '{source_type}'")
        source_counts[source_type] += 1

        quality_tier = chunk["quality_tier"]
        if quality_tier not in ALLOWED_TIERS:
            errors.append(f"{where}: invalid quality_tier '{quality_tier}'")
        tier_counts[quality_tier] += 1

        topic_tags = chunk["topic_tags"]
        if not isinstance(topic_tags, list) or not topic_tags:
            errors.append(f"{where}: topic_tags must be a non-empty list")
        elif len(topic_tags) > 3:
            errors.append(f"{where}: topic_tags should be 1-3 tags")

        text = chunk["text"]
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{where}: text must be non-empty string")
            continue

        min_words = MIN_WORDS_BY_SOURCE.get(source_type, 100)
        if chunk["word_count"] < min_words:
            errors.append(
                f"{where}: word_count {chunk['word_count']} is below minimum {min_words} for {source_type}"
            )

        if is_pure_announcement(text):
            errors.append(f"{where}: looks like pure announcement content")

        if not is_startup_related(text):
            errors.append(f"{where}: appears off-topic for startup/YC corpus")

        measured_wc = count_words(text)
        if abs(measured_wc - chunk["word_count"]) > 3:
            errors.append(
                f"{where}: word_count mismatch stored={chunk['word_count']} measured={measured_wc}"
            )

    print("Phase 1 Validation Report")
    print("-" * 28)
    print(f"File: {file_path}")
    print(f"Total chunks: {len(data)}")
    print(f"Source counts: {dict(source_counts)}")
    print(f"Tier counts: {dict(tier_counts)}")

    if errors:
        print(f"\nFAILED: {len(errors)} issue(s) found")
        for err in errors[:50]:
            print(f"- {err}")
        if len(errors) > 50:
            print(f"- ... {len(errors) - 50} more")
        return 1

    print("\nPASSED: all checks succeeded")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 1 chunks.json")
    parser.add_argument(
        "--file",
        default="data/processed/chunks.json",
        help="Path to chunks JSON file",
    )
    args = parser.parse_args()
    raise SystemExit(validate(Path(args.file)))


if __name__ == "__main__":
    main()

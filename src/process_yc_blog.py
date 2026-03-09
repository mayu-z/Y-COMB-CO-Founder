import argparse
import csv
import hashlib
import os
import re
from collections import Counter
from pathlib import Path


def clean_header(key: str) -> str:
    return key.strip().lstrip("\ufeff").strip('"')


def sanitize_filename(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "", name)
    safe = re.sub(r"\s+", " ", safe).strip()
    if not safe:
        safe = "untitled"
    return safe[:140]


def extract_title(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            if "| Y Combinator" in candidate:
                candidate = candidate.replace("| Y Combinator", "").strip()
            return candidate
    return "Untitled"


def extract_author(text: str) -> str:
    match = re.search(
        r"\bby\s+([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’\-.,\s]{1,100}?)(?=\d|\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        author = re.sub(r"\s+", " ", match.group(1)).strip(" ,.-")
        if author:
            return author
    return "Y Combinator"


def url_is_blog_post(url: str) -> tuple[bool, str | None]:
    lower = url.lower().strip()

    if "/blog/" not in lower:
        return False, "url_not_blog_post"
    if "/blog/tag/" in lower:
        return False, "tag_page"
    if lower.endswith("/blog") or lower.endswith("/blog/"):
        return False, "blog_index"

    return True, None


def process_csv(input_csv: Path, output_dir: Path) -> None:
    csv.field_size_limit(10 ** 7)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = Counter()

    with input_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV appears to have no header row")

        original_fieldnames = reader.fieldnames
        cleaned_fieldnames = [clean_header(name) for name in original_fieldnames]
        reader.fieldnames = cleaned_fieldnames

        for row in reader:
            summary["rows_total"] += 1

            normalized_row = {
                clean_header(str(key)): (value or "")
                for key, value in row.items()
            }

            url = normalized_row.get("url", "").strip().strip('"')
            text = normalized_row.get("text", "").strip()

            if not url:
                summary["skipped_missing_url"] += 1
                continue
            if not text:
                summary["skipped_missing_text"] += 1
                continue

            keep, reason = url_is_blog_post(url)
            if not keep:
                summary[f"skipped_{reason}"] += 1
                continue

            word_count = len(text.split())
            if word_count < 300:
                summary["skipped_too_short"] += 1
                continue

            title = extract_title(text)
            author = extract_author(text)

            url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
            filename = f"{sanitize_filename(title)}_{url_hash}.txt"
            output_path = output_dir / filename

            payload = (
                f"Title: {title}\n"
                f"Author: {author}\n"
                f"URL: {url}\n\n"
                f"{text.strip()}\n"
            )

            try:
                output_path.write_text(payload, encoding="utf-8")
                summary["saved_posts"] += 1
            except OSError:
                summary["skipped_write_error"] += 1

    print("YC Blog Processing Summary")
    print("-" * 28)
    print(f"Input CSV: {input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Total rows read: {summary['rows_total']}")
    print(f"Posts saved: {summary['saved_posts']}")

    print("\nSkipped rows by reason:")
    skip_keys = sorted(key for key in summary if key.startswith("skipped_"))
    if not skip_keys:
        print("- none")
    else:
        for key in skip_keys:
            print(f"- {key.replace('skipped_', '')}: {summary[key]}")


def main() -> None:
    default_input = Path("data/raw/yc_blog/dataset_website-content-crawler_2026-03-08_15-10-16-923.csv")
    default_output = Path("data/raw/yc_blog")

    parser = argparse.ArgumentParser(description="Process Apify YC blog CSV into per-post text files")
    parser.add_argument("--input", type=Path, default=default_input, help="Path to Apify CSV")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Directory for output .txt files")
    args = parser.parse_args()

    process_csv(args.input, args.output_dir)


if __name__ == "__main__":
    main()

"""
YC Co-Founder Data Scraper — Phase 1
=====================================
Sources:
  1. Dalton Caldwell + Michael Seibel YouTube videos (via transcripts)
  2. YC official YouTube channel videos
  3. YC Startup Library (ycombinator.com/library)
  4. Paul Graham essays (paulgraham.com)
  5. HN threads about YC applications / rejections
  6. Medium / Substack founder posts about YC rejection → acceptance

Output: data/raw/<source>/<id>.json
Each file shape:
  {
    "id": str,
    "source": str,
    "type": "transcript" | "essay" | "post" | "thread",
    "title": str,
    "url": str,
    "content": str,
    "metadata": { ... }
  }

Install before running:
  pip install yt-dlp youtube-transcript-api requests beautifulsoup4 praw
"""

import os
import re
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse, quote_plus

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
RAW_DIR    = BASE_DIR / "data" / "raw"
STATS_FILE = BASE_DIR / "data" / "scrape_stats.json"

for folder in ["youtube", "yc_library", "pg_essays", "hn_threads", "founder_posts"]:
    (RAW_DIR / folder).mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def save(folder: str, doc_id: str, data: dict):
    path = RAW_DIR / folder / f"{doc_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"  saved → {path.relative_to(BASE_DIR)}")

def slug(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    return text[:max_len]

def get(url: str, delay: float = 1.5, **kwargs) -> requests.Response | None:
    time.sleep(delay)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20, **kwargs)
        r.raise_for_status()
        return r
    except Exception as e:
        log.warning(f"  GET failed {url}: {e}")
        return None

def already_scraped(folder: str, doc_id: str) -> bool:
    return (RAW_DIR / folder / f"{doc_id}.json").exists()

# ── 1. YouTube transcripts ───────────────────────────────────────────────────

# Curated list — Dalton, Michael Seibel, YC official.
# Add more video IDs here anytime.
YC_YOUTUBE_VIDEOS = {
    # Dalton Caldwell
    "B5tU2447OK8": "How to Apply and Succeed at YC — Dalton Caldwell (Startup School)",
    "8yiOcCPvyNE": "How to Apply and Succeed at YC — Dalton Caldwell",
    "8pNxKX1SUGE": "All About Pivoting — Dalton Caldwell",
    "dlfjs_eEEzs": "Co-Founder Mistakes That Kill Companies — Dalton & Michael",
    "wnyI7ZM_Mrk": "Understanding Investor Terms & Incentives — Dalton & Michael",
    "UqKzpLqXuI0": "Why Investors Can't Fix Your Company — Dalton & Michael",
    "hQSzFMDQTH8": "How to Get Your First Customers — YC",
    "z1iF1c8w5Lg": "How to Talk to Users — YC",
    # Michael Seibel
    "PkDtWZjpfE4": "Michael Seibel — How to Build an MVP",
    "6KGvkNsYmRs": "Michael Seibel — Making Something People Want",
    "0LNQxU0LoDg": "Michael Seibel — How to Talk to Investors",
    # Gustaf Alstromer — growth
    "6lY9ACFR3as": "How to Get Your First Users — Gustaf Alstromer",
    # YC — application / interview tips
    "W4aVkXUIKGg": "YC Interview Prep — Common Mistakes",
    "uVOe-al布": "Startup School — How to Evaluate Startup Ideas",
    "DOIXFZBqLwA": "Startup School — How to Plan an MVP",
    # Rejection / resilience
    "OwJa1Cf9DRDU": "Dalton & Michael — Founders Ok With Rejection",
}

# Playlists to crawl (we fetch the playlist page and extract video IDs)
YC_PLAYLISTS = [
    "PLQ-uHSnFig5Nd98Sc9I-kkc0ZWe8peRMC",  # Dalton & Michael
    "PLQ-uHSnFig5MaGi8jFwLITJTBtcuBhntB",  # Startup School 2024
    "PLQ-uHSnFig5OYPCpQiQ2oYMSIXCpHkXtT",  # How to Apply
]

def fetch_playlist_video_ids(playlist_id: str) -> list[str]:
    """
    Fetch video IDs from a YouTube playlist page.
    Falls back gracefully — returns [] if blocked.
    """
    url = f"https://www.youtube.com/playlist?list={playlist_id}"
    r = get(url, delay=2)
    if not r:
        return []
    ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', r.text)
    unique = list(dict.fromkeys(ids))  # preserve order, deduplicate
    log.info(f"  playlist {playlist_id}: found {len(unique)} video IDs")
    return unique

def fetch_transcript(video_id: str) -> str | None:
    """
    Fetch YouTube transcript using youtube-transcript-api.
    Returns plain text or None.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join(seg["text"] for seg in transcript).strip()
    except ImportError:
        log.warning("  youtube-transcript-api not installed. Run: pip install youtube-transcript-api")
        return None
    except Exception as e:
        log.warning(f"  transcript unavailable for {video_id}: {e}")
        return None

def scrape_youtube(extra_ids: list[str] | None = None):
    log.info("=== SOURCE 1: YouTube transcripts ===")

    # Gather all video IDs
    all_ids: dict[str, str] = dict(YC_YOUTUBE_VIDEOS)

    # Crawl playlists
    for pid in YC_PLAYLISTS:
        for vid in fetch_playlist_video_ids(pid):
            if vid not in all_ids:
                all_ids[vid] = ""   # title unknown until we scrape

    if extra_ids:
        for vid in extra_ids:
            all_ids[vid] = ""

    log.info(f"  total videos to process: {len(all_ids)}")
    stats = {"attempted": 0, "saved": 0, "skipped": 0, "failed": 0}

    for video_id, known_title in all_ids.items():
        stats["attempted"] += 1
        if already_scraped("youtube", video_id):
            log.info(f"  skip (exists): {video_id}")
            stats["skipped"] += 1
            continue

        transcript = fetch_transcript(video_id)
        if not transcript:
            stats["failed"] += 1
            continue

        title = known_title or f"YC Video {video_id}"
        save("youtube", video_id, {
            "id": video_id,
            "source": "youtube",
            "type": "transcript",
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "content": transcript,
            "metadata": {
                "word_count": len(transcript.split()),
                "scraped_at": datetime.utcnow().isoformat(),
            },
        })
        stats["saved"] += 1

    log.info(f"  YouTube done: {stats}")
    return stats

# ── 2. YC Startup Library ────────────────────────────────────────────────────

YC_LIBRARY_BASE = "https://www.ycombinator.com"
YC_LIBRARY_SEARCH = "https://www.ycombinator.com/library"

# Categories most relevant to rejection / application / advice
YC_LIBRARY_CATEGORIES = [
    "Applying to YC",
    "How to Pitch",
    "Founder Stories",
    "Fundraising",
    "Building Product",
    "Startup School",
]

def scrape_yc_library():
    log.info("=== SOURCE 2: YC Startup Library ===")
    stats = {"attempted": 0, "saved": 0, "skipped": 0, "failed": 0}

    # Fetch all library items (paginated JSON API)
    items = []
    page = 1
    while True:
        url = f"{YC_LIBRARY_SEARCH}?page={page}"
        r = get(url, delay=2)
        if not r:
            break

        soup = BeautifulSoup(r.text, "html.parser")

        # Extract Next.js __NEXT_DATA__ JSON — richest source of structured data
        script = soup.find("script", id="__NEXT_DATA__")
        if script:
            try:
                data = json.loads(script.string)
                posts = (
                    data.get("props", {})
                        .get("pageProps", {})
                        .get("posts", [])
                )
                if not posts:
                    break
                items.extend(posts)
                log.info(f"  page {page}: +{len(posts)} items")
                page += 1
                continue
            except Exception:
                pass

        # Fallback: look for article links
        links = soup.select("a[href^='/library/']")
        if not links:
            break
        for a in links:
            href = a.get("href", "")
            if href and href not in [i.get("url", "") for i in items]:
                items.append({"url": YC_LIBRARY_BASE + href, "title": a.get_text(strip=True)})
        page += 1
        if page > 20:
            break

    log.info(f"  found {len(items)} library items total")

    for item in items:
        stats["attempted"] += 1
        url  = item.get("url") or (YC_LIBRARY_BASE + item.get("slug", ""))
        title = item.get("title", "")
        doc_id = slug(title or urlparse(url).path)

        if already_scraped("yc_library", doc_id):
            stats["skipped"] += 1
            continue

        r = get(url, delay=2)
        if not r:
            stats["failed"] += 1
            continue

        soup = BeautifulSoup(r.text, "html.parser")

        # Try to grab the main article body
        content = ""
        for selector in ["article", "main", ".prose", ".content", "section"]:
            tag = soup.select_one(selector)
            if tag:
                content = tag.get_text(separator="\n", strip=True)
                break
        if not content:
            content = soup.get_text(separator="\n", strip=True)

        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        if len(content) < 200:
            stats["failed"] += 1
            continue

        save("yc_library", doc_id, {
            "id": doc_id,
            "source": "yc_library",
            "type": "post",
            "title": title,
            "url": url,
            "content": content,
            "metadata": {
                "word_count": len(content.split()),
                "scraped_at": datetime.utcnow().isoformat(),
            },
        })
        stats["saved"] += 1

    log.info(f"  YC Library done: {stats}")
    return stats

# ── 3. Paul Graham essays ────────────────────────────────────────────────────

PG_INDEX = "https://paulgraham.com/articles.html"

# These essays are most relevant to YC / startups / rejection
PG_PRIORITY = {
    "startupideas", "growth", "ds", "startupmistakes", "investors",
    "fundraising", "convince", "determination", "genius", "schlep",
    "ambitious", "founder", "notnot", "before", "hiring", "equity",
    "corpdev", "cities", "startuplessons", "market", "jessica",
    "opia", "early", "credentials", "hw", "badeconomy", "die",
}

def scrape_pg_essays():
    log.info("=== SOURCE 3: Paul Graham essays ===")
    stats = {"attempted": 0, "saved": 0, "skipped": 0, "failed": 0}

    r = get(PG_INDEX, delay=1)
    if not r:
        log.warning("  Could not reach paulgraham.com")
        return stats

    soup = BeautifulSoup(r.text, "html.parser")
    links = [
        (a.get_text(strip=True), urljoin(PG_INDEX, a["href"]))
        for a in soup.find_all("a", href=True)
        if a["href"].endswith(".html") and "/" not in a["href"]
    ]
    log.info(f"  found {len(links)} PG essay links")

    for title, url in links:
        stats["attempted"] += 1
        essay_id = urlparse(url).path.lstrip("/").replace(".html", "")
        priority = essay_id in PG_PRIORITY

        if already_scraped("pg_essays", essay_id):
            stats["skipped"] += 1
            continue

        r = get(url, delay=1.2 if priority else 2.0)
        if not r:
            stats["failed"] += 1
            continue

        soup = BeautifulSoup(r.text, "html.parser")

        # PG's site is very simple — content is in <font> or <table>
        content = ""
        for tag in soup.find_all(["font", "p", "td"]):
            text = tag.get_text(separator="\n", strip=True)
            if len(text) > len(content):
                content = text

        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        if len(content) < 300:
            stats["failed"] += 1
            continue

        save("pg_essays", essay_id, {
            "id": essay_id,
            "source": "pg_essays",
            "type": "essay",
            "title": title,
            "url": url,
            "content": content,
            "metadata": {
                "priority": priority,
                "word_count": len(content.split()),
                "scraped_at": datetime.utcnow().isoformat(),
            },
        })
        stats["saved"] += 1

    log.info(f"  PG essays done: {stats}")
    return stats

# ── 4. Hacker News threads ───────────────────────────────────────────────────

HN_SEARCH = "https://hn.algolia.com/api/v1/search"

HN_QUERIES = [
    "rejected by YC got in",
    "YC rejection what changed",
    "applied YC multiple times accepted",
    "YC application tips",
    "why YC rejected us",
    "got into YC after rejection",
    "YC interview rejection",
    "YC application video pitch",
    "YC co-founder advice",
    "Dalton Caldwell YC advice",
]

def fetch_hn_item(item_id: int) -> dict | None:
    url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
    r = get(url, delay=0.5)
    if not r:
        return None
    return r.json()

def clean_hn_html(html_text: str) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator="\n", strip=True)

def scrape_hn_thread(story: dict) -> str:
    """Recursively collect story + comments into one text blob."""
    parts = []
    title = story.get("title", "")
    text  = clean_hn_html(story.get("text", ""))
    url   = story.get("url", "")

    if title:
        parts.append(f"TITLE: {title}")
    if url:
        parts.append(f"URL: {url}")
    if text:
        parts.append(f"POST:\n{text}")

    # Fetch top-level comments (not recursing deeper to keep it clean)
    for kid_id in (story.get("kids") or [])[:40]:
        item = fetch_hn_item(kid_id)
        if not item or item.get("deleted") or item.get("dead"):
            continue
        comment = clean_hn_html(item.get("text", ""))
        author  = item.get("by", "anon")
        if comment and len(comment) > 30:
            parts.append(f"\n[{author}]: {comment}")

    return "\n".join(parts)

def scrape_hn():
    log.info("=== SOURCE 4: Hacker News threads ===")
    stats = {"attempted": 0, "saved": 0, "skipped": 0, "failed": 0}
    seen_ids: set[int] = set()

    for query in HN_QUERIES:
        log.info(f"  HN query: '{query}'")
        params = {
            "query": query,
            "tags": "(story,ask_hn)",
            "hitsPerPage": 20,
            "minPoints": 5,       # filter out low-signal posts
        }
        r = get(HN_SEARCH, delay=1.5, params=params)
        if not r:
            continue

        hits = r.json().get("hits", [])
        log.info(f"    → {len(hits)} hits")

        for hit in hits:
            story_id = int(hit.get("objectID", 0))
            if not story_id or story_id in seen_ids:
                continue
            seen_ids.add(story_id)
            stats["attempted"] += 1

            doc_id = str(story_id)
            if already_scraped("hn_threads", doc_id):
                stats["skipped"] += 1
                continue

            # Fetch full item from HN API for comments
            story = fetch_hn_item(story_id)
            if not story:
                stats["failed"] += 1
                continue

            content = scrape_hn_thread(story)
            if len(content) < 100:
                stats["failed"] += 1
                continue

            save("hn_threads", doc_id, {
                "id": doc_id,
                "source": "hn_threads",
                "type": "thread",
                "title": story.get("title", hit.get("title", "")),
                "url": f"https://news.ycombinator.com/item?id={story_id}",
                "content": content,
                "metadata": {
                    "points": story.get("score", 0),
                    "comments": story.get("descendants", 0),
                    "author": story.get("by", ""),
                    "query": query,
                    "scraped_at": datetime.utcnow().isoformat(),
                },
            })
            stats["saved"] += 1

    log.info(f"  HN done: {stats}")
    return stats

# ── 5. Founder posts (Medium + Substack) ────────────────────────────────────

# These are manually curated high-signal posts + search-based discovery.
# Medium blocks scraping aggressively, so we use their public RSS / partner API
# where possible and fall back to direct fetch.

FOUNDER_SEED_URLS = [
    # Known high-signal posts about YC rejection → acceptance
    "https://blog.ycombinator.com/the-yc-application/",
    "https://blog.ycombinator.com/yc-application-advice/",
    "https://blog.ycombinator.com/how-to-apply-to-yc/",
    # Add any specific Medium/Substack URLs you find here:
    # "https://medium.com/@founder/how-we-got-into-yc-after-3-rejections",
]

MEDIUM_SEARCH_TAGS = [
    "https://medium.com/tag/y-combinator/latest",
    "https://medium.com/tag/startup-funding/latest",
    "https://medium.com/tag/yc-application/latest",
]

def scrape_founder_posts():
    log.info("=== SOURCE 5: Founder posts ===")
    stats = {"attempted": 0, "saved": 0, "skipped": 0, "failed": 0}

    urls_to_scrape: list[tuple[str, str]] = []

    # Add seed URLs
    for url in FOUNDER_SEED_URLS:
        urls_to_scrape.append((url, "seed"))

    # Discover from Medium tag pages
    for tag_url in MEDIUM_SEARCH_TAGS:
        log.info(f"  crawling Medium tag: {tag_url}")
        r = get(tag_url, delay=3)
        if not r:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "medium.com" in href and "/@" in href and len(href) > 40:
                urls_to_scrape.append((href.split("?")[0], "medium_tag"))

    # Deduplicate
    seen = set()
    unique_urls = []
    for url, source in urls_to_scrape:
        if url not in seen:
            seen.add(url)
            unique_urls.append((url, source))

    log.info(f"  total founder post URLs to try: {len(unique_urls)}")

    for url, src in unique_urls:
        stats["attempted"] += 1
        doc_id = slug(urlparse(url).path.replace("/", "_"))
        if not doc_id:
            doc_id = slug(url)

        if already_scraped("founder_posts", doc_id):
            stats["skipped"] += 1
            continue

        r = get(url, delay=3)
        if not r:
            stats["failed"] += 1
            continue

        soup = BeautifulSoup(r.text, "html.parser")

        # Try common article selectors
        content = ""
        title   = ""
        for sel in ["article", "main", ".postArticle-content", "section.metabar~div"]:
            tag = soup.select_one(sel)
            if tag and len(tag.get_text()) > 300:
                content = tag.get_text(separator="\n", strip=True)
                break
        if not content:
            content = soup.get_text(separator="\n", strip=True)

        title_tag = soup.find("h1") or soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        # Filter out low-relevance pages
        yc_keywords = ["y combinator", "yc", "rejection", "startup", "founder", "pitch"]
        if not any(kw in content.lower() for kw in yc_keywords):
            stats["failed"] += 1
            continue

        if len(content) < 400:
            stats["failed"] += 1
            continue

        save("founder_posts", doc_id, {
            "id": doc_id,
            "source": "founder_posts",
            "type": "post",
            "title": title,
            "url": url,
            "content": content,
            "metadata": {
                "discovery_source": src,
                "word_count": len(content.split()),
                "scraped_at": datetime.utcnow().isoformat(),
            },
        })
        stats["saved"] += 1

    log.info(f"  Founder posts done: {stats}")
    return stats

# ── Runner ───────────────────────────────────────────────────────────────────

def run_all(sources: list[str] | None = None):
    all_sources = {
        "youtube":       scrape_youtube,
        "yc_library":    scrape_yc_library,
        "pg_essays":     scrape_pg_essays,
        "hn":            scrape_hn,
        "founder_posts": scrape_founder_posts,
    }
    to_run = {k: v for k, v in all_sources.items()
              if sources is None or k in sources}

    all_stats = {}
    start = time.time()

    for name, fn in to_run.items():
        log.info(f"\n{'='*50}")
        try:
            all_stats[name] = fn()
        except Exception as e:
            log.error(f"  {name} crashed: {e}", exc_info=True)
            all_stats[name] = {"error": str(e)}

    elapsed = round(time.time() - start, 1)
    total_saved = sum(s.get("saved", 0) for s in all_stats.values() if isinstance(s, dict))

    summary = {
        "run_at": datetime.utcnow().isoformat(),
        "elapsed_seconds": elapsed,
        "total_saved": total_saved,
        "by_source": all_stats,
    }

    with open(STATS_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\n{'='*50}")
    log.info(f"DONE — {total_saved} documents saved in {elapsed}s")
    log.info(f"Stats written to {STATS_FILE}")
    return summary

# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YC Co-Founder Phase 1 Scraper")
    parser.add_argument(
        "--sources", nargs="*",
        choices=["youtube", "yc_library", "pg_essays", "hn", "founder_posts"],
        help="Which sources to scrape (default: all)",
    )
    parser.add_argument(
        "--youtube-ids", nargs="*", metavar="VIDEO_ID",
        help="Extra YouTube video IDs to add",
    )
    args = parser.parse_args()

    run_all(sources=args.sources)
"""
Phase 0 — Data Collection

All-in-one scraper for the YC Co-Founder RAG project.
Collects data from 6 sources:
  1. Paul Graham Essays       (requests + BS4)
  2. YC Blog Posts             (requests + BS4)
  3. Startup School Lectures   (youtube-transcript-api)
  4. YC Company Directory      (Kaggle CSV download)
  5. Hacker News Threads       (HN Algolia API)
  6. YC Application Questions  (manual file creation)

Usage:
  python src/scraper.py pg          # Paul Graham essays
  python src/scraper.py yc_blog     # YC Blog posts
  python src/scraper.py startup     # Startup School transcripts
  python src/scraper.py companies   # YC Company directory
  python src/scraper.py hn          # Hacker News threads
  python src/scraper.py yc_app      # YC application questions
  python src/scraper.py all         # Run everything
"""

import csv
import json
import os
import re
import sys
import time
import argparse
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

PG_DIR = os.path.join(DATA_DIR, "pg_essays")
YC_BLOG_DIR = os.path.join(DATA_DIR, "yc_blog")
STARTUP_SCHOOL_DIR = os.path.join(DATA_DIR, "startup_school")
COMPANIES_CSV = os.path.join(DATA_DIR, "companies.csv")
HN_FILE = os.path.join(DATA_DIR, "hn_threads.json")
YC_APP_FILE = os.path.join(DATA_DIR, "yc_application_questions.txt")

REQUEST_DELAY = 2  # seconds between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def sanitize_filename(title: str) -> str:
    """Turn a title into a safe, readable filename."""
    name = re.sub(r'[<>:"/\\|?*]', "", title)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > 120:
        name = name[:120].rsplit(" ", 1)[0]
    return name


def fetch_page(url: str) -> str | None:
    """Fetch a page and return its HTML, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        return resp.text
    except requests.RequestException as exc:
        print(f"  ⚠ Failed to fetch {url}: {exc}")
        return None


def fetch_json(url: str, params: dict = None) -> dict | None:
    """Fetch JSON from an API endpoint."""
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"  ⚠ Failed to fetch {url}: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE 1: Paul Graham Essays
# ═══════════════════════════════════════════════════════════════════════════
PG_BASE_URL = "https://paulgraham.com/"
PG_ARTICLES_URL = "https://paulgraham.com/articles.html"

PG_PRIORITY = [
    "Do Things That Don't Scale",
    "How to Get Startup Ideas",
    "Default Alive or Default Dead",
    "Hiring is Obsolete",
    "How to Convince Investors",
    "What Investors Look For",
]


def get_pg_essay_links() -> list[tuple[str, str]]:
    """Parse the articles index page and return (title, url) pairs."""
    html = fetch_page(PG_ARTICLES_URL)
    if html is None:
        print("❌ Could not fetch the PG articles index page.")
        return []

    soup = BeautifulSoup(html, "lxml")
    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        title = a.get_text(strip=True)
        if not title or href.startswith("#") or href.startswith("mailto:"):
            continue
        if any(href.lower().endswith(ext) for ext in (".png", ".jpg", ".gif", ".jpeg")):
            continue
        full_url = urljoin(PG_BASE_URL, href)
        if "paulgraham.com" not in full_url or full_url in seen:
            continue
        seen.add(full_url)
        links.append((title, full_url))

    print(f"📋 Found {len(links)} PG essay links.")
    return links


def extract_pg_essay_text(html: str) -> str:
    """Extract clean essay text from a PG essay page."""
    soup = BeautifulSoup(html, "lxml")

    # Replace <br> with newlines, then unwrap inline tags
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag_name in ["i", "b", "em", "strong", "font", "span", "u", "a"]:
        for tag in soup.find_all(tag_name):
            tag.unwrap()

    # Find main content in widest <td>
    candidates = soup.find_all("td", width=True)
    best = ""
    for td in candidates:
        text = td.get_text()
        if len(text) > len(best):
            best = text

    if len(best) < 200:
        body = soup.find("body")
        if body:
            best = body.get_text()

    # Clean up blank lines
    lines = best.split("\n")
    cleaned, blank_count = [], 0
    for line in lines:
        s = line.strip()
        if s == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(s)
    return "\n".join(cleaned).strip()


def scrape_pg_essays(limit: int | None = None):
    """Scrape Paul Graham essays and save as .txt files."""
    print("\n🚀 Source 1: Paul Graham Essays")
    os.makedirs(PG_DIR, exist_ok=True)

    links = get_pg_essay_links()
    if limit:
        links = links[:limit]

    total = len(links)
    saved = skipped = failed = 0

    for i, (title, url) in enumerate(links, 1):
        filename = sanitize_filename(title) + ".txt"
        filepath = os.path.join(PG_DIR, filename)

        if os.path.exists(filepath):
            print(f"  [{i}/{total}] ⏭ Already exists: {filename}")
            skipped += 1
            continue

        print(f"  [{i}/{total}] 📥 {title}")
        html = fetch_page(url)
        if not html:
            failed += 1
            continue

        text = extract_pg_essay_text(html)
        if len(text) < 100:
            print(f"           ⚠ Too short ({len(text)} chars), skipping.")
            failed += 1
            continue

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        saved += 1
        print(f"           ✅ Saved ({len(text):,} chars)")

        if i < total:
            time.sleep(REQUEST_DELAY)

    print(f"\n{'='*50}")
    print(f"✅ PG Essays — Saved: {saved} | Skipped: {skipped} | Failed: {failed}")

    # Priority check
    if os.path.exists(PG_DIR) and os.listdir(PG_DIR):
        print("\n🔍 Priority essay check:")
        files = os.listdir(PG_DIR)
        for name in PG_PRIORITY:
            found = any(name.lower() in f.lower() for f in files)
            print(f"  {'✅' if found else '❌ NOT FOUND'} — {name}")


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE 2: YC Blog
# ═══════════════════════════════════════════════════════════════════════════
YC_BLOG_BASE = "https://www.ycombinator.com/blog"
YC_BLOG_RSS_URL = "https://www.ycombinator.com/blog/rss/"

# Focus on posts by key YC partners
YC_PARTNERS = [
    "dalton caldwell", "michael seibel", "jared friedman",
    "gustaf alströmer", "gustaf alstromer",
]


def get_yc_blog_posts_rss() -> list[dict]:
    """
    Fetch YC blog posts via RSS feed.
    The main blog page is JS-rendered, but the RSS feed works with plain requests.
    """
    import xml.etree.ElementTree as ET

    rss_url = YC_BLOG_RSS_URL
    print(f"  📋 Fetching RSS feed from {rss_url}...")

    # Explicit RSS health check
    try:
        rss_resp = requests.get(rss_url, headers=HEADERS, timeout=15)
        content_type = rss_resp.headers.get("Content-Type", "")
        print(
            f"  ℹ RSS check: status={rss_resp.status_code}, "
            f"content_type={content_type}, url={rss_resp.url}"
        )
        if rss_resp.status_code != 200:
            print("  ❌ RSS feed URL appears invalid (non-200).")
            return []
    except requests.RequestException as exc:
        print(f"  ❌ RSS feed check failed: {exc}")
        return []

    html = fetch_page(rss_url)
    if not html:
        print("  ❌ Could not fetch RSS feed.")
        return []

    try:
        root = ET.fromstring(html)
    except ET.ParseError as e:
        print(f"  ❌ Failed to parse RSS: {e}")
        return []

    posts = []
    # RSS structure: rss > channel > item
    channel = root.find("channel")
    if channel is None:
        return []

    for item in channel.findall("item"):
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        creator = item.findtext("{http://purl.org/dc/elements/1.1/}creator", "").strip()
        # Content might be in content:encoded
        content = item.findtext(
            "{http://purl.org/rss/1.0/modules/content/}encoded", ""
        ).strip()
        description = item.findtext("description", "").strip()

        if title and link:
            posts.append({
                "title": title,
                "url": link,
                "date": pub_date,
                "author": creator,
                "content_html": content or description,
            })

    print(f"📋 Found {len(posts)} YC blog posts from RSS.")
    return posts


def get_yc_blog_posts_direct(max_pages: int = 15) -> list[dict]:
    """
    Fallback: scrape ycombinator.com/blog directly with pagination.
    """
    print(f"  🔁 Falling back to direct blog scrape with pagination (max_pages={max_pages})...")

    post_urls: list[str] = []
    seen_urls: set[str] = set()

    for page in range(1, max_pages + 1):
        page_url = YC_BLOG_BASE if page == 1 else f"{YC_BLOG_BASE}?page={page}"
        html = fetch_page(page_url)
        if not html:
            if page == 1:
                print("  ❌ Could not fetch YC blog page 1.")
            break

        soup = BeautifulSoup(html, "lxml")
        new_on_page = 0

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            full_url = urljoin(YC_BLOG_BASE, href)
            if "/blog/" not in full_url:
                continue
            if full_url.endswith("/blog") or full_url.endswith("/blog/"):
                continue
            if full_url.endswith("/blog/rss") or "/blog/rss/" in full_url:
                continue
            if "?" in full_url:
                # Keep canonical URL without query string
                full_url = full_url.split("?", 1)[0]
            if full_url in seen_urls:
                continue

            seen_urls.add(full_url)
            post_urls.append(full_url)
            new_on_page += 1

        print(f"    • Page {page}: found {new_on_page} new post links")
        if page > 2 and new_on_page == 0:
            break
        time.sleep(0.5)

    posts: list[dict] = []
    for idx, post_url in enumerate(post_urls, 1):
        page_html = fetch_page(post_url)
        if not page_html:
            continue

        soup = BeautifulSoup(page_html, "lxml")

        # Extract metadata where available
        title = ""
        title_tag = soup.find("meta", property="og:title")
        if title_tag and title_tag.get("content"):
            title = title_tag["content"].strip()
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()

        author = ""
        author_tag = soup.find("meta", attrs={"name": "author"})
        if author_tag and author_tag.get("content"):
            author = author_tag["content"].strip()

        date = ""
        date_tag = soup.find("meta", property="article:published_time")
        if date_tag and date_tag.get("content"):
            date = date_tag["content"].strip()

        # Capture main content section for downstream extraction
        content_html = ""
        for sel in ["article", "main", '[class*="content"]', '[class*="post"]', '[class*="blog"]']:
            el = soup.select_one(sel)
            if el and len(el.get_text(strip=True)) > 200:
                content_html = str(el)
                break
        if not content_html and soup.body:
            content_html = str(soup.body)

        if not title:
            title = post_url.rstrip("/").split("/")[-1].replace("-", " ").title()

        posts.append({
            "title": title,
            "url": post_url,
            "date": date,
            "author": author,
            "content_html": content_html,
        })

        if idx % 25 == 0:
            print(f"    • Parsed {idx}/{len(post_urls)} direct post pages")

    print(f"📋 Found {len(posts)} YC blog posts from direct pagination scrape.")
    return posts


def extract_yc_blog_text(html_content: str) -> str:
    """Extract clean text from YC blog post HTML content."""
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "lxml")

    # Replace <br> with newlines
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Unwrap inline tags
    for tag_name in ["i", "b", "em", "strong", "font", "span", "u"]:
        for tag in soup.find_all(tag_name):
            tag.unwrap()

    text = soup.get_text(separator="\n")

    # Clean up blank lines
    lines = text.split("\n")
    cleaned, blank_count = [], 0
    for line in lines:
        s = line.strip()
        if s == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(s)

    return "\n".join(cleaned).strip()


def scrape_yc_blog(limit: int | None = None):
    """Scrape YC Blog posts via RSS and save as .txt files."""
    print("\n🚀 Source 2: YC Blog")
    os.makedirs(YC_BLOG_DIR, exist_ok=True)

    posts = get_yc_blog_posts_rss()
    if not posts:
        posts = get_yc_blog_posts_direct(max_pages=20)

    if limit:
        posts = posts[:limit]

    total = len(posts)
    saved = skipped = failed = 0
    duplicate_skipped = 0
    passed_300 = 0
    short_skipped = 0

    for i, post in enumerate(posts, 1):
        title = post["title"]
        url = post["url"]
        author = post.get("author", "")
        date = post.get("date", "")

        filename = sanitize_filename(title) + ".txt"
        filepath = os.path.join(YC_BLOG_DIR, filename)

        if os.path.exists(filepath):
            print(f"  [{i}/{total}] ⏭ Already exists: {filename}")
            skipped += 1
            duplicate_skipped += 1
            continue

        print(f"  [{i}/{total}] 📥 {title}")

        # Try RSS content first, fall back to fetching full page
        text = extract_yc_blog_text(post.get("content_html", ""))

        if len(text.split()) < 100:
            # RSS content truncated — fetch full page
            page_html = fetch_page(url)
            if page_html:
                page_soup = BeautifulSoup(page_html, "lxml")
                # Look for article/main content
                for sel in ["article", "main", '[class*="content"]',
                            '[class*="post"]', '[class*="blog"]']:
                    el = page_soup.select_one(sel)
                    if el and len(el.get_text(strip=True)) > 200:
                        text = extract_yc_blog_text(str(el))
                        break

        # Skip short posts (< 300 words)
        word_count = len(text.split())
        if word_count < 300:
            print(f"           ⚠ Too short ({word_count} words), skipping.")
            failed += 1
            short_skipped += 1
            continue

        passed_300 += 1

        # Prepend metadata header
        header = f"Title: {title}\n"
        if author:
            header += f"Author: {author}\n"
        if date:
            header += f"Date: {date}\n"
        header += f"URL: {url}\n"
        header += "---\n\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + text)
        saved += 1
        print(f"           ✅ Saved ({word_count} words, author: {author or 'unknown'})")

        if i < total:
            time.sleep(REQUEST_DELAY)

    print(f"\n{'='*50}")
    print(f"✅ YC Blog — Saved: {saved} | Skipped: {skipped} | Failed: {failed}")
    print(f"  ℹ Debug — RSS/direct posts considered: {total}")
    print(f"  ℹ Debug — Passed 300-word filter: {passed_300}")
    print(f"  ℹ Debug — Skipped as duplicates: {duplicate_skipped}")
    print(f"  ℹ Debug — Skipped as too short: {short_skipped}")


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE 3: Startup School Lectures (YouTube Transcripts)
# ═══════════════════════════════════════════════════════════════════════════

# Key Startup School lectures with their YouTube video IDs
STARTUP_SCHOOL_LECTURES = [
    {"title": "How to Talk to Users", "speaker": "Gustaf Alströmer", "video_id": "z1iF1c8w5Lg"},
    {"title": "How to Find Product Market Fit", "speaker": "David Rusenko", "video_id": "0LNQxT9LvM0"},
    {"title": "How to Build and Sell", "speaker": "Michael Seibel", "video_id": "f9_-HFkzTfw"},
    {"title": "Fundraising Fundamentals", "speaker": "Kirsty Nathoo", "video_id": "WpCRBiJHnKw"},
    {"title": "How to Apply to YC", "speaker": "Dalton Caldwell", "video_id": "9eChMSraS2E"},
    {"title": "How to Plan an MVP", "speaker": "Michael Seibel", "video_id": "1hHMwLxN6EM"},
    {"title": "How to Get Your First Customers", "speaker": "Gustaf Alströmer", "video_id": "hyYCn_kAngI"},
    {"title": "How to Pitch Your Startup", "speaker": "Michael Seibel", "video_id": "17XZGUX_9iM"},
    {"title": "How to Get and Test Startup Ideas", "speaker": "Michael Seibel", "video_id": "vDXkpJlhIBg"},
    {"title": "How to Split Equity Among Co-Founders", "speaker": "Michael Seibel", "video_id": "ETOyE_1IWUQ"},
    {"title": "Startup Mechanics", "speaker": "Kirsty Nathoo", "video_id": "ufEIBOOYwWA"},
    {"title": "Nine Business Models", "speaker": "Dalton Caldwell", "video_id": "1MTRVfm8UHQ"},
    {"title": "How to Launch (Again and Again)", "speaker": "Kat Manalac", "video_id": "3xU050kMbHM"},
    {"title": "How to Work Together", "speaker": "Tim Brady", "video_id": "30a5yFBd7Fo"},
    {"title": "How to Set KPIs and Goals", "speaker": "Gustaf Alströmer", "video_id": "lSjDLZrhw4c"},
    {"title": "Understanding SAFEs and Priced Equity Rounds", "speaker": "Kirsty Nathoo", "video_id": "yilx3y4RjfA"},
    {"title": "How to Improve Conversion Rates", "speaker": "Gustaf Alströmer", "video_id": "PGqX9fpweyc"},
    {"title": "Startup School Welcome", "speaker": "Gary Tan", "video_id": "O1FjUcvM-TQ"},
    {"title": "How to Prioritize Your Time", "speaker": "Adora Cheung", "video_id": "XcCmMOWuAF4"},
    {"title": "How to Evaluate Startup Ideas", "speaker": "Kevin Hale", "video_id": "DOtCl5PU8F0"},
    # Additional YC YouTube channel videos (10)
    {"title": "YC Channel Talk 01", "speaker": "Y Combinator", "video_id": "DNSXlBmukck"},
    {"title": "YC Channel Talk 02", "speaker": "Y Combinator", "video_id": "UPGB-hsAoVY"},
    {"title": "YC Channel Talk 03", "speaker": "Y Combinator", "video_id": "Q8wVMdwhlh4"},
    {"title": "YC Channel Talk 04", "speaker": "Y Combinator", "video_id": "PQU9o_5rHC4"},
    {"title": "YC Channel Talk 05", "speaker": "Y Combinator", "video_id": "rWUWfj_PqmM"},
    {"title": "YC Channel Talk 06", "speaker": "Y Combinator", "video_id": "4uzGDAoNOZc"},
    {"title": "YC Channel Talk 07", "speaker": "Y Combinator", "video_id": "qwmmWzPnhog"},
    {"title": "YC Channel Talk 08", "speaker": "Y Combinator", "video_id": "K5JoLAauzq4"},
    {"title": "YC Channel Talk 09", "speaker": "Y Combinator", "video_id": "leQ89XSHILw"},
    {"title": "YC Channel Talk 10", "speaker": "Y Combinator", "video_id": "dC_3ys349bU"},
]

# Filler words to clean from transcripts
FILLER_WORDS = ["um", "uh", "you know", "like,", "so,", "right,"]


def clean_transcript(text: str) -> str:
    """Clean filler words from a transcript."""
    for filler in FILLER_WORDS:
        # Case-insensitive removal
        pattern = re.compile(re.escape(filler), re.IGNORECASE)
        text = pattern.sub("", text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    return text.strip()


def scrape_startup_school(limit: int | None = None):
    """Fetch YouTube transcripts for Startup School lectures."""
    print("\n🚀 Source 3: Startup School Lectures")
    os.makedirs(STARTUP_SCHOOL_DIR, exist_ok=True)

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("❌ youtube-transcript-api not installed. Run:")
        print("   pip install youtube-transcript-api")
        return

    lectures = STARTUP_SCHOOL_LECTURES
    if limit:
        lectures = lectures[:limit]

    print(f"  ℹ Startup lecture IDs configured: {len(lectures)}")

    total = len(lectures)
    saved = skipped = failed = 0
    failed_lectures: list[dict] = []
    ytt_api = YouTubeTranscriptApi()

    for i, lecture in enumerate(lectures, 1):
        title = lecture["title"]
        speaker = lecture["speaker"]
        video_id = lecture["video_id"]

        filename = sanitize_filename(f"{speaker} - {title}") + ".txt"
        filepath = os.path.join(STARTUP_SCHOOL_DIR, filename)

        if os.path.exists(filepath):
            print(f"  [{i}/{total}] ⏭ Already exists: {filename}")
            skipped += 1
            continue

        print(f"  [{i}/{total}] 📥 {title} — {speaker}")

        try:
            transcript = ytt_api.fetch(video_id)

            # Combine all text segments
            full_text = " ".join(
                snippet.text for snippet in transcript.snippets
            )

            # Clean filler words
            full_text = clean_transcript(full_text)

            if len(full_text) < 100:
                print(f"           ⚠ Transcript too short, skipping.")
                failed += 1
                continue

            # Add metadata header
            header = f"Title: {title}\n"
            header += f"Speaker: {speaker}\n"
            header += f"Source: https://youtube.com/watch?v={video_id}\n"
            header += "---\n\n"

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(header + full_text)
            saved += 1
            word_count = len(full_text.split())
            print(f"           ✅ Saved ({word_count:,} words)")

        except Exception as exc:
            print(f"           ⚠ Failed: {exc}")
            failed += 1
            failed_lectures.append({
                "title": title,
                "speaker": speaker,
                "video_id": video_id,
                "error": str(exc),
            })

        if i < total:
            time.sleep(1)

    print(f"\n{'='*50}")
    print(f"✅ Startup School — Saved: {saved} | Skipped: {skipped} | Failed: {failed}")
    if failed_lectures:
        print("  ℹ Failed transcript fetches:")
        for item in failed_lectures:
            print(f"    - {item['title']} ({item['video_id']})")


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE 4: YC Company Directory
# ═══════════════════════════════════════════════════════════════════════════

YC_COMPANIES_API = "https://api.ycombinator.com/v0.1/companies"


def scrape_yc_companies(limit: int | None = None):
    """
    Fetch YC company data from the YC API and save as CSV.
    The YC website has a public JSON API we can use instead of scraping.
    """
    print("\n🚀 Source 4: YC Company Directory")
    os.makedirs(os.path.dirname(COMPANIES_CSV), exist_ok=True)

    if os.path.exists(COMPANIES_CSV):
        print(f"  ⏭ Already exists: {COMPANIES_CSV}")
        return

    all_companies = []
    page = 0

    while True:
        print(f"  📋 Fetching companies page {page + 1}...")
        data = fetch_json(
            YC_COMPANIES_API,
            params={"page": page, "per_page": 100}
        )

        if not data or "companies" not in data:
            break

        companies = data["companies"]
        if not companies:
            break

        for c in companies:
            # Skip stealth companies with no description
            desc = (c.get("oneLiner") or c.get("one_liner") or "").strip()
            if not desc:
                continue

            # Determine status from badges
            badges = c.get("badges", [])
            status = "Active"
            if "Acquired" in badges:
                status = "Acquired"
            elif "Inactive" in badges:
                status = "Inactive"

            all_companies.append({
                "name": c.get("name", ""),
                "batch": c.get("batchName") or c.get("batch", ""),
                "industry": ", ".join(c.get("industries", c.get("tags", []))),
                "region": c.get("region", ""),
                "one_liner": desc,
                "website": c.get("website") or c.get("url", ""),
                "status": status,
                "team_size": c.get("teamSize") or c.get("team_size", ""),
            })

        page += 1
        time.sleep(0.5)  # API is generous but be polite

        if limit and len(all_companies) >= limit:
            all_companies = all_companies[:limit]
            break

        # Safety: YC has ~5000 companies, 50 pages should cover it
        if page > 60:
            break

    if not all_companies:
        print("  ⚠ No companies fetched. The API may have changed.")
        print("  💡 Alternative: Download from Kaggle YC dataset.")
        return

    # Write CSV
    fields = ["name", "batch", "industry", "region", "one_liner",
              "website", "status", "team_size"]
    with open(COMPANIES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_companies)

    print(f"\n{'='*50}")
    print(f"✅ YC Companies — Saved {len(all_companies)} companies to {COMPANIES_CSV}")


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE 5: Hacker News Threads (Algolia API)
# ═══════════════════════════════════════════════════════════════════════════

HN_ALGOLIA_API = "https://hn.algolia.com/api/v1"

HN_SEARCH_QUERIES = [
    "YC application",
    "YC interview",
    "got into YC",
    "rejected by YC",
    "Y Combinator experience",
    "YC batch",
    "applying to YC",
]


def fetch_hn_threads(query: str, min_points: int = 10, max_pages: int = 5) -> list[dict]:
    """Search HN for threads matching a query with enough upvotes."""
    threads = []

    for page in range(max_pages):
        url = f"{HN_ALGOLIA_API}/search"
        params = {
            "query": query,
            "tags": "(story,ask_hn)",
            "hitsPerPage": 50,
            "page": page,
            "numericFilters": f"points>={min_points}",
        }
        data = fetch_json(url, params)
        if not data or "hits" not in data:
            break

        hits = data["hits"]
        if not hits:
            break

        for hit in hits:
            threads.append({
                "title": hit.get("title", ""),
                "url": hit.get("url", ""),
                "author": hit.get("author", ""),
                "points": hit.get("points", 0),
                "num_comments": hit.get("num_comments", 0),
                "created_at": hit.get("created_at", ""),
                "objectID": hit.get("objectID", ""),
                "query": query,
                "story_text": hit.get("story_text", ""),
            })

        time.sleep(0.5)

    return threads


def fetch_launch_hn_posts(min_points: int = 10, max_pages: int = 10) -> list[dict]:
    """Fetch Launch HN posts — founders describing their YC-backed startups."""
    posts = []

    for page in range(max_pages):
        url = f"{HN_ALGOLIA_API}/search"
        params = {
            "query": "Launch HN",
            "tags": "story",
            "hitsPerPage": 50,
            "page": page,
            "numericFilters": f"points>={min_points}",
        }
        data = fetch_json(url, params)
        if not data or "hits" not in data:
            break

        hits = data["hits"]
        if not hits:
            break

        for hit in hits:
            title = hit.get("title", "")
            if "Launch HN" in title or "launch hn" in title.lower():
                posts.append({
                    "title": title,
                    "url": hit.get("url", ""),
                    "author": hit.get("author", ""),
                    "points": hit.get("points", 0),
                    "num_comments": hit.get("num_comments", 0),
                    "created_at": hit.get("created_at", ""),
                    "objectID": hit.get("objectID", ""),
                    "query": "Launch HN",
                    "story_text": hit.get("story_text", ""),
                })

        time.sleep(0.5)

    return posts


def scrape_hn_threads(limit: int | None = None):
    """Fetch HN threads about YC and save as JSON."""
    print("\n🚀 Source 5: Hacker News Threads")
    os.makedirs(os.path.dirname(HN_FILE), exist_ok=True)

    if os.path.exists(HN_FILE):
        print(f"  ⏭ Already exists: {HN_FILE}")
        return

    all_threads = []

    # Search for YC-related threads
    for query in HN_SEARCH_QUERIES:
        print(f"  🔍 Searching: \"{query}\"")
        threads = fetch_hn_threads(query, min_points=10)
        all_threads.extend(threads)
        print(f"     Found {len(threads)} threads")

    # Fetch Launch HN posts
    print(f"  🔍 Fetching Launch HN posts...")
    launch_posts = fetch_launch_hn_posts(min_points=10)
    all_threads.extend(launch_posts)
    print(f"     Found {len(launch_posts)} Launch HN posts")

    # Deduplicate by objectID
    seen_ids = set()
    unique = []
    for t in all_threads:
        oid = t["objectID"]
        if oid not in seen_ids:
            seen_ids.add(oid)
            unique.append(t)

    if limit:
        unique = unique[:limit]

    # Save as JSON
    with open(HN_FILE, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"✅ HN Threads — Saved {len(unique)} threads to {HN_FILE}")


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE 6: YC Application Questions
# ═══════════════════════════════════════════════════════════════════════════

YC_APPLICATION_QUESTIONS = """YC Application Questions
========================

These are the standard Y Combinator application questions that founders must answer
when applying to a YC batch. They are publicly known and widely discussed.

COMPANY
-------

1. Company name:
2. Company URL, if any:
3. If you have a demo, what's the URL?
4. Describe what your company does in 50 characters or less.
5. What is your company going to make? Please describe your product and what it does or will do.
6. Where do you live now, and where would the company be based after YC?

FOUNDERS
--------

7. Founders: Please enter the url of a 1 minute unlisted YouTube video introducing the founder(s).
8. Please tell us about an interesting project, preferably outside of class or work, that two or more of you created together.
9. How long have the founders known one another and how did you meet? Have any of the founders not met in person?

PROGRESS
--------

10. How far along are you? Do you have a beta, a+
 prototype, or just an idea?
11. How long have each of you been working on this? How much of that has been full-time?
12. Are people using your product?
13. Do you have revenue? If so, how much and what is your monthly growth rate?
14. If you are applying with the same idea as a previous batch, did anything change? If you applied with a different idea, why did you pivot and what did you learn from the last idea?
15. If you have already participated or committed to participate in an ideaincubator, "deck day", or "demo day" event, please tell us about it.

IDEA
----

16. Why did you pick this idea to work on? Do you have domain expertise in this area? How do you know people need what you're making?
17. What's new about what you're making? What substitutes do people resort to because it doesn't exist yet (or they don't know about it)?
18. Who are your competitors, and who might become competitors? Who do you fear most?
19. What do you understand about your business that other companies in it just don't get?

EQUITY
------

20. Have you incorporated, or formed any legal entity (like an LLC) yet?
21. Have you taken any investment yet?
22. If you have not formed the company yet, describe the planned equity ownership breakdown among the founders, employees and any other proposed stockholders.

LEGAL
-----

23. Are any of the founders covered by noncompetes or intellectual property agreements that overlap with your project? If so, please explain.
24. Who writes code, or does other technical work on your product? Was any of it done by a non-founder? Please explain.

OTHERS
------

25. Is there anything else we should know about your company?
26. If you had any other ideas you considered applying with, please list them — one may be something we've been waiting to fund.

CURIOUS
-------

27. Please tell us something surprising or amusing that one of you has discovered.
"""


def create_yc_application_questions():
    """Create the YC application questions file."""
    print("\n🚀 Source 6: YC Application Questions")
    os.makedirs(os.path.dirname(YC_APP_FILE), exist_ok=True)

    if os.path.exists(YC_APP_FILE):
        print(f"  ⏭ Already exists: {YC_APP_FILE}")
        return

    with open(YC_APP_FILE, "w", encoding="utf-8") as f:
        f.write(YC_APPLICATION_QUESTIONS.strip())

    print(f"  ✅ Created {YC_APP_FILE}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
SOURCES = {
    "pg": ("Paul Graham Essays", scrape_pg_essays),
    "yc_blog": ("YC Blog", scrape_yc_blog),
    "startup": ("Startup School", scrape_startup_school),
    "companies": ("YC Companies", scrape_yc_companies),
    "hn": ("HN Threads", scrape_hn_threads),
    "yc_app": ("YC Application Questions", create_yc_application_questions),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 0 — Data Collection for YC Co-Founder RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Sources: " + ", ".join(SOURCES.keys()) + ", all",
    )
    parser.add_argument(
        "source",
        choices=list(SOURCES.keys()) + ["all"],
        help="Which data source to scrape",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit items to scrape (for testing)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 0 — Data Collection")
    print("=" * 60)

    if args.source == "all":
        for key, (name, func) in SOURCES.items():
            if key == "yc_app":
                func()  # no limit argument
            else:
                func(limit=args.limit)
    else:
        name, func = SOURCES[args.source]
        if args.source == "yc_app":
            func()
        else:
            func(limit=args.limit)

    print("\n" + "=" * 60)
    print("  🏁 Done!")
    print("=" * 60)

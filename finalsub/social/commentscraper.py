"""
YouTube Comment Scraper for Hindi NLP corpus
Scrapes comments from specified video IDs and saves to CSV + plain text.
"""

import csv
import os
import sys
import time
from datetime import datetime

try:
    from yt_dlp import YoutubeDL
except ImportError:
    print("ERROR: yt_dlp not installed. Run: pip install yt-dlp")
    sys.exit(1)

VIDEO_IDS = [
    "MPALBVeaiho",
    "xhGwYrE0ezc",
    "5ebeS877-QE",
    "Hgx5hN0OT4w",
    "p1MK6_eyevY",
]

MAX_COMMENTS_PER_VIDEO = 20000
OUTPUT_DIR = "youtube_comments"


def scrape_comments(video_id: str, max_comments: int) -> list[dict]:
    """Extract comments from a single YouTube video using yt-dlp."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    comments = []

    ydl_opts = {
        "skip_download": True,
        "writecomments": True,
        "getcomments": True,
        "extractor_args": {
            "youtube": {
                "max_comments": [str(max_comments)],
                "comment_sort": ["top"],
            }
        },
        "quiet": True,
        "no_warnings": True,
    }

    print(f"  Fetching comments for video: {video_id}")
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            video_title = info.get("title", "Unknown")
            raw_comments = info.get("comments", []) or []

            for c in raw_comments[:max_comments]:
                comments.append({
                    "video_id":    video_id,
                    "video_title": video_title,
                    "comment_id":  c.get("id", ""),
                    "author":      c.get("author", ""),
                    "text":        c.get("text", "").strip(),
                    "likes":       c.get("like_count", 0),
                    "timestamp":   c.get("timestamp", ""),
                    "is_reply":    c.get("parent", "") != "root",
                })

            print(f"  ✓ '{video_title}' — {len(comments)} comments collected")

        except Exception as e:
            print(f"  ✗ Failed for {video_id}: {e}")

    return comments


def save_csv(all_comments: list[dict], path: str):
    fields = ["video_id", "video_title", "comment_id", "author",
              "text", "likes", "timestamp", "is_reply"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_comments)
    print(f"\n✓ CSV saved: {path}  ({len(all_comments)} total comments)")


def save_plain_text(all_comments: list[dict], path: str):
    """One comment per line — easy to feed into your NLP pipeline."""
    with open(path, "w", encoding="utf-8") as f:
        for c in all_comments:
            text = c["text"].replace("\n", " ").strip()
            if text:
                f.write(text + "\n")
    print(f"✓ Plain text saved: {path}")


def print_summary(all_comments: list[dict]):
    from collections import Counter
    print("\n── Summary ──────────────────────────────")
    counts = Counter(c["video_id"] for c in all_comments)
    for vid, n in counts.items():
        print(f"  {vid}: {n} comments")
    print(f"  Total: {len(all_comments)} comments")

    # quick script detection — useful for your Devanagari vs Roman split
    devanagari = sum(
        1 for c in all_comments
        if any("\u0900" <= ch <= "\u097f" for ch in c["text"])
    )
    roman = len(all_comments) - devanagari
    print(f"\n  Script breakdown (approximate):")
    print(f"    Devanagari-containing : {devanagari}")
    print(f"    Mostly Roman/Hinglish : {roman}")
    print("─────────────────────────────────────────")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_comments = []
    for vid in VIDEO_IDS:
        if vid.startswith("VIDEO_ID"):
            print(f"  ⚠ Skipping placeholder: {vid}")
            continue
        comments = scrape_comments(vid, MAX_COMMENTS_PER_VIDEO)
        all_comments.extend(comments)
        time.sleep(2)   # polite delay between requests

    if not all_comments:
        print("\nNo comments collected. Check your video IDs.")
        return

    csv_path  = os.path.join(OUTPUT_DIR, f"comments_{timestamp}.csv")
    txt_path  = os.path.join(OUTPUT_DIR, f"comments_{timestamp}.txt")

    save_csv(all_comments, csv_path)
    save_plain_text(all_comments, txt_path)
    print_summary(all_comments)


if __name__ == "__main__":
    main()
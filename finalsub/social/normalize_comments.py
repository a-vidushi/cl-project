import argparse
import csv
import re
import unicodedata
from pathlib import Path

# ── optional deps with graceful fallback ──────────────────────────────────────
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠  pandas not found — CSV input disabled. Run: uv add pandas")

# ─────────────────────────────────────────────────────────────────────────────
# Script / character ranges
# ─────────────────────────────────────────────────────────────────────────────
DEVANAGARI   = re.compile(r'[\u0900-\u097F]')
GURMUKHI     = re.compile(r'[\u0A00-\u0A7F]')   # Punjabi
ARABIC_URDU  = re.compile(r'[\u0600-\u06FF]')
LATIN        = re.compile(r'[A-Za-z]')
EMOJI        = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002600-\U000027BF"  # misc symbols
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U00002702-\U000027B0"  # dingbats
    "\U0000200D"             # zero-width joiner
    "\U00002640-\U00002642"  # gender symbols
    "]+",
    flags=re.UNICODE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Spam / noise patterns
# ─────────────────────────────────────────────────────────────────────────────
SPAM_PATTERNS = [
    re.compile(r'https?://\S+'),                        # URLs
    re.compile(r'www\.\S+'),
    re.compile(r'@\w+'),                                # mentions
    re.compile(r'#\w+'),                                # hashtags
    re.compile(r'\b(download|install|subscribe|click|link|app|apk)\b',
               re.IGNORECASE),
    re.compile(r'market\s*wolf', re.IGNORECASE),        # ad seen in corpus
    re.compile(r'\d{10,}'),                             # phone numbers
]

# Repeated character normalizer — "bahutttt" → "bahutt" (keep max 2)
REPEATED_CHARS = re.compile(r'(.)\1{2,}')

# Repeated word normalizer — "good good good" → "good good"
REPEATED_WORDS = re.compile(r'\b(\w+)(\s+\1){2,}\b', re.IGNORECASE)

# Poorna viram variants → sentence boundary marker
POORNA_VIRAM  = re.compile(r'[।॥]')

# Non-linguistic punctuation bursts
PUNCT_BURST   = re.compile(r'[!?.,;:]{3,}')


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def detect_script(text: str) -> str:
    deva  = len(DEVANAGARI.findall(text))
    latin = len(LATIN.findall(text))
    total = deva + latin

    if total == 0:
        return 'other'

    deva_ratio  = deva  / total
    latin_ratio = latin / total

    if deva_ratio  >= 0.70:
        return 'devanagari'
    if latin_ratio >= 0.70:
        return 'roman'
    if deva_ratio  >= 0.25 and latin_ratio >= 0.25:
        return 'mixed'
    return 'other'


def is_noise(text: str) -> tuple[bool, str]:
    """
    Returns (True, reason) if comment should be dropped, else (False, '').
    """
    stripped = EMOJI.sub('', text).strip()

    # too short after emoji removal
    if len(stripped) < 10:
        return True, 'too_short'

    # purely numeric / punctuation
    if re.fullmatch(r'[\d\s\W]+', stripped):
        return True, 'non_linguistic'

    # spam keyword hit
    for pat in SPAM_PATTERNS:
        if pat.search(text):
            return True, 'spam'

    # script is Gurmukhi or Arabic (out of scope for Hindi pipeline)
    gurmukhi_ratio = len(GURMUKHI.findall(text)) / max(len(text), 1)
    arabic_ratio   = len(ARABIC_URDU.findall(text)) / max(len(text), 1)
    hindi_ratio = len(DEVANAGARI.findall(text)) / max(len(text), 1)
    latin_ratio = len(LATIN.findall(text)) / max(len(text), 1)
    if (hindi_ratio + latin_ratio) < 0.7:
        return True, 'not_valid'

    return False, ''


def normalize_text(text: str) -> str:
    """
    Apply text normalization steps in order.
    """
    # 1. Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)

    # 2. Remove URLs, mentions, hashtags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)

    # 3. Remove emojis
    text = EMOJI.sub(' ', text)

    # 4. Poorna viram → period
    text = POORNA_VIRAM.sub('.', text)

    # 5. Collapse punctuation bursts
    text = PUNCT_BURST.sub(lambda m: m.group()[0], text)

    # 6. Normalize repeated characters (hahaha → haha, bahuuuut → bahuut)
    text = REPEATED_CHARS.sub(r'\1\1', text)

    # 7. Normalize repeated words (good good good → good good)
    text = REPEATED_WORDS.sub(r'\1 \1', text)

    # 8. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_comments(path: Path) -> list[str]:
    """Load comments from .txt (one per line) or .csv (uses 'text' column)."""
    suffix = path.suffix.lower()

    if suffix == '.txt':
        lines = path.read_text(encoding='utf-8').splitlines()
        return [l.strip() for l in lines if l.strip()]

    if suffix == '.csv':
        if not HAS_PANDAS:
            raise RuntimeError("pandas required for CSV input. Run: uv add pandas")
        df = pd.read_csv(path)
        # try common column names
        for col in ('text', 'comment', 'body', 'content'):
            if col in df.columns:
                return df[col].dropna().astype(str).tolist()
        # fallback: first column
        return df.iloc[:, 0].dropna().astype(str).tolist()

    raise ValueError(f"Unsupported file type: {suffix}. Use .txt or .csv")


def run_pipeline(comments: list[str]) -> list[dict]:
    """
    Run the full normalization pipeline.
    Returns a list of record dicts.
    """
    seen   = set()
    records = []

    for raw in comments:
        raw = raw.strip()
        if not raw:
            continue

        # ── deduplicate ──
        if raw in seen:
            continue
        seen.add(raw)

        # ── noise filter ──
        noisy, reason = is_noise(raw)
        if noisy:
            records.append({
                'raw':        raw,
                'normalized': '',
                'script':     '',
                'status':     f'dropped:{reason}',
            })
            continue

        # ── normalize ──
        norm = normalize_text(raw)

        # ── post-normalization length check ──
        if len(norm) < 4:
            records.append({
                'raw':        raw,
                'normalized': norm,
                'script':     '',
                'status':     'dropped:empty_after_norm',
            })
            continue

        # ── script detection ──
        script = detect_script(norm)

        records.append({
            'raw':        raw,
            'normalized': norm,
            'script':     script,
            'status':     'kept',
        })

    return records


def save_outputs(records: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── full CSV ──
    csv_path = out_dir / 'normalized_comments.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['raw', 'normalized', 'script', 'status'])
        writer.writeheader()
        writer.writerows(records)

    # ── per-script plain text (kept comments only) ──
    kept = [r for r in records if r['status'] == 'kept']
    for script in ('devanagari', 'roman', 'mixed'):
        subset = [r['normalized'] for r in kept if r['script'] == script]
        out_path = out_dir / f'comments_{script}.txt'
        out_path.write_text('\n'.join(subset), encoding='utf-8')

    # ── pipeline report ──
    total   = len(records)
    kept_n  = sum(1 for r in records if r['status'] == 'kept')
    dropped = total - kept_n

    from collections import Counter
    script_counts  = Counter(r['script'] for r in kept)
    drop_reasons   = Counter(
        r['status'].replace('dropped:', '')
        for r in records if r['status'] != 'kept'
    )

    print("\n── Normalization Report ─────────────────────────")
    print(f"  Input comments   : {total}")
    print(f"  Kept             : {kept_n}  ({kept_n/total*100:.1f}%)")
    print(f"  Dropped          : {dropped}  ({dropped/total*100:.1f}%)")
    print(f"\n  Script breakdown (kept):")
    for script, n in script_counts.most_common():
        print(f"    {script:<14}: {n}")
    print(f"\n  Drop reasons:")
    for reason, n in drop_reasons.most_common():
        print(f"    {reason:<25}: {n}")
    print(f"\n  Output → {out_dir.resolve()}")
    print("─────────────────────────────────────────────────\n")

    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Normalize Hindi social media comments for NLP pipeline."
    )
    parser.add_argument(
        '--input',  '-i', required=True,
        help='Path to input file (.txt one-per-line, or .csv with text column)'
    )
    parser.add_argument(
        '--output', '-o', default='normalized',
        help='Output directory (default: ./normalized/)'
    )
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}")
        return

    print(f"Loading comments from {in_path} ...")
    comments = load_comments(in_path)
    print(f"  {len(comments)} raw comments loaded")

    print("Running normalization pipeline ...")
    records = run_pipeline(comments)

    save_outputs(records, out_path)


if __name__ == '__main__':
    main()
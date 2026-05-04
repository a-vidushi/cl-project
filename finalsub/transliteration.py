import re
import unicodedata
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from unicode_converter import unicode_converter

def load_transcripts(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8-sig") as fh:
        text = fh.read()
    
    # Treat entire file as ONE episode
    return {"DATA": text}


# WEEK 1 – CORPUS EXPLORATION & UTF-8 VALIDATION

POORNA_VIRAM = '\u0964'   # ।  – Hindi full stop
DBL_DANDA    = '\u0965'   # ॥  – double danda

def week1_analysis(episodes: dict[str, str]) -> dict:
    """
    Validate UTF-8, characterise the corpus, and locate sentence boundaries.

    Returns a report dict with per-episode and aggregate statistics.
    """
    report = {}

    for label, text in episodes.items():
        bad_chars = [c for c in text if unicodedata.category(c) == 'Cs']

        dev_chars   = sum(1 for c in text if '\u0900' <= c <= '\u097f')
        latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        digits      = sum(1 for c in text if c.isdigit())
        spaces      = sum(1 for c in text if c.isspace())
        punc_chars  = sum(1 for c in text if not c.isalnum() and not c.isspace())

        sentence_pattern = r'[' + POORNA_VIRAM + DBL_DANDA + r'.?!]'
        sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]

        tokens = text.split()
        dev_tokens  = [t for t in tokens if any('\u0900' <= c <= '\u097f' for c in t)]
        latin_tokens = [t for t in tokens if all(c.isascii() for c in t if c.isalpha()) and
                        any(c.isalpha() for c in t)]

        report[label] = {
            "total_chars":    len(text),
            "devanagari_chars": dev_chars,
            "latin_chars":    latin_chars,
            "digit_chars":    digits,
            "invalid_unicode": len(bad_chars),
            "total_tokens":   len(tokens),
            "devanagari_tokens": len(dev_tokens),
            "latin_tokens":   len(latin_tokens),
            "sentence_count": len(sentences),
            "poorna_viram_count": text.count(POORNA_VIRAM),
            "dbl_danda_count": text.count(DBL_DANDA),
        }

    # Aggregate
    report["TOTAL"] = {
        k: sum(v[k] for v in report.values() if isinstance(v.get(k), int))
        for k in next(iter(report.values())).keys()
    }
    return report


# WEEK 2 – PREPROCESSING & TRANSLITERATION

def preprocess(text: str) -> str:
    """
    Stage 1 preprocessing (applied before transliteration):
      • NFC normalisation (canonical decomposition → recomposition)
      • Replace poorna viram (।) and double danda (॥) with ASCII full stop so
        that downstream regex rules see uniform sentence boundaries.
      • Normalise non-breaking spaces and other whitespace variants to ASCII space.
      • Collapse multiple consecutive spaces / newlines to single space.
    Latin-script tokens (acronyms, borrowings) are preserved untouched.
    """
    # 1. NFC normalisation
    text = unicodedata.normalize("NFC", text)

    # 2. Punctuation normalisation
    text = text.replace(POORNA_VIRAM, '.')
    text = text.replace(DBL_DANDA, '.')
    text = text.replace('\u2013', '-')   # en-dash → hyphen
    text = text.replace('\u2014', ' ')   # em-dash → space
    text = text.replace('\u2018', "'")   # left single quote
    text = text.replace('\u2019', "'")   # right single quote / apostrophe
    text = text.replace('\u201c', '"')
    text = text.replace('\u201d', '"')

    # 3. Whitespace normalisation
    text = text.replace('\xa0', ' ')     # non-breaking space
    text = re.sub(r'[ \t]+', ' ', text)  # multiple spaces → one
    text = re.sub(r'\n{3,}', '\n\n', text)  # 3+ newlines → 2

    return text.strip()


def transliterate_mixed(text: str) -> str:
    # Regex: Devanagari block vs non-Devanagari block
    seg_re = re.compile(r'([\u0900-\u097f\u0964-\u0965]+|[^\u0900-\u097f\u0964-\u0965]+)')

    def transliterate_token(tok: str) -> str:
        parts = []
        for seg in seg_re.findall(tok):
            if any('\u0900' <= c <= '\u097f' for c in seg):
                parts.append(unicode_converter(seg))
            else:
                parts.append(seg)
        return ''.join(parts)

    # Preserve inter-token whitespace by splitting on whitespace runs
    tokens = re.split(r'(\s+)', text)
    result = []
    for tok in tokens:
        if re.match(r'\s+', tok):
            result.append(tok)
        else:
            result.append(transliterate_token(tok))
    return ''.join(result)


def week2_pipeline(episodes: dict[str, str]) -> dict[str, str]:
    return {
        label: transliterate_mixed(preprocess(text))
        for label, text in episodes.items()
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python transliteration.py <input_file> <output_file>")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2]

    # Step 1: Load data
    episodes = load_transcripts(input_file)

    # Step 2: Week 1 analysis
    report = week1_analysis(episodes)
    print("=== WEEK 1 REPORT ===")
    for k, v in report.items():
        print(k, v)

    # Step 3: Week 2 processing
    processed = week2_pipeline(episodes)

    # Save processed output
    with open(output_file, "w", encoding="utf-8") as f:
        for label, text in processed.items():
            f.write(text)

    print(f"\nTransliterated text saved to {output_file}")
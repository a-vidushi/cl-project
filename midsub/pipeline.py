import re
import unicodedata
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from unicode_converter import unicode_converter

def load_transcripts(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()

    # re.split with a capturing group interleaves labels into the result list:
    # [pre, label1, body1, label2, body2, label3, body3]
    parts = re.split(r'\*\*(February 2026|January 2026|December 2025)\*\*', raw)
    if len(parts) > 1:
        # parts[1], parts[3], parts[5] are the labels; [2],[4],[6] are the bodies
        labels  = parts[1::2]
        bodies  = parts[2::2]
        return {lbl.strip(): body for lbl, body in zip(labels, bodies)}
    else:
        filename = os.path.basename(path)
        return {filename: raw}



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


# WEEK 3 – REGEX PATTERNS: REDUPLICATION, ACRONYMS, HYPHENATED PAIRS

# 1. Reduplication 

REDUPLICATIONS_RE = re.compile(
    r'\b(\w{2,})-\1\b',
    re.UNICODE | re.IGNORECASE
)


def find_reduplications(text: str) -> list[str]:
    return REDUPLICATIONS_RE.findall(text)   # returns the captured group (the word)


# 2. Acronyms 

ACRONYM_RE = re.compile(r'(?<!-)\b[A-Z]{2,}\b')


def find_acronyms(text: str) -> list[str]:
    return ACRONYM_RE.findall(text)


# 3. Hyphenated pairs (non-reduplication) 

HYPHEN_PAIR_RE = re.compile(
    r'\b(\w{2,})-(\w{2,})\b',
    re.UNICODE
)


def find_hyphenated_pairs(text: str) -> list[tuple[str, str]]:
    matches = []
    for left, right in HYPHEN_PAIR_RE.findall(text):
        # Exclude reduplications
        if left.lower() != right.lower():
            matches.append((left, right))
    return matches


# WEEK 4 – REGEX PATTERNS: CONJUNCTIVE VERBS, COMPOUND VERBS,
#           HONORIFICS, MULTI-WORD EXPRESSIONS

    # 4. Conjunctive Verbs (absolutives / gerunds) 

CONJUNCTIVE_VERB_RE = re.compile(
    r'(?<!\w)'
    r'([a-zāīūēōṭḍṇṣśṛṅñṁḥr̥]{3,})'
    r'-?'
    r'(kar|kē|ke)'
    r'(?!\w)',
    re.UNICODE
)


def find_conjunctive_verbs(text: str) -> list[tuple[str, str]]:
    return CONJUNCTIVE_VERB_RE.findall(text)


# 5. Compound Verbs 

_VECTOR = (
    r'(?:lē|liyā?|lī)'            # lēnā  family: lē / liyā / lī
    r'|(?:dē|diyā?|dī)'           # dēnā  family: dē / diyā / dī
    r'|(?:jā|gayā|gaī|gaē|jātā)'  # jānā  family
    r'|(?:ā|ātā|āī|āyā)'          # ānā   family
    r'|(?:paṛ|paṛā|paṛī|paṛē)'   # paṛnā family
)

COMPOUND_VERB_RE = re.compile(
    r'\b(\w{2,})\b'
    r'(?:\s+\S+){0,2}\s+'
    r'\b(' + _VECTOR + r')\b',
    re.UNICODE
)

_NON_VERB_STEMS = {
    "kā", "kī", "kē", "kō", "mēṃ", "par", "sē", "nē",
    "yah", "vah", "is", "us", "in", "un", "ēk", "aur",
    "yē", "vē", "kuch", "sab", "jō", "tō", "hī", "bhī",
    "na", "nahīṃ", "ab", "tab", "jab", "phir", "ik",
}


def find_compound_verbs(text: str) -> list[tuple[str, str]]:
    matches = []
    for stem, vec in COMPOUND_VERB_RE.findall(text):
        if stem.lower() not in _NON_VERB_STEMS:
            matches.append((stem, vec))
    return matches


# 6. Honorifics 

HONORIFIC_RE = re.compile(
    r'\b(āp|tum|tū)\b',
    re.UNICODE
)
_REGISTER = {
    "āp":  "formal",
    "tum": "familiar",
    "tū":  "intimate",
}


def find_honorifics(text: str) -> list[dict]:
    results = []
    for m in HONORIFIC_RE.finditer(text):
        form = m.group(1)
        results.append({
            "form":     form,
            "register": _REGISTER.get(form, "unknown"),
            "start":    m.start(),
            "end":      m.end(),
        })
    return results


# 7. Multi-Word Expressions (MWEs) 

MWE_LEXICON: list[tuple[str, str]] = [
    ("kē bād",       "after"),
    ("kē āgē",       "ahead of / before"),
    ("kē bīc",       "between / among"),
    ("kē sāth",      "with / along with"),
    ("kē liyē",      "for / in order to"),
    ("kē bārē mē",   "about / regarding"),
    ("kē tahat",     "under / pursuant to"),
    ("kī vajah sē",  "because of"),
    ("kī taraf",     "towards"),
    ("kī jagah",     "instead of"),
    ("kī or",        "towards"),
    ("kē rūp mē",    "in the form of / as"),
    ("kē dvārā",     "by means of"),
    ("kē anusār",    "according to"),
    ("kē alāvā",     "apart from / besides"),
    ("sē lēkar",     "from … up to"),
    ("kē sāmn",      "in front of"),
    ("kē nīcē",      "below / under"),
    ("kē ūpar",      "above / on top of"),
    ("isī liyē",     "that is why"),
    ("is liyē",      "therefore / so"),
    ("jis tarah",    "in the same way as"),
    ("is tarah",     "in this way"),
    ("nā sirf",      "not only"),
    ("balki",        "but also"),
]


_MWE_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r'\b' + re.escape(mwe) + r'\b', re.UNICODE | re.IGNORECASE),
     mwe, gloss)
    for mwe, gloss in MWE_LEXICON
]


def find_mwes(text: str) -> list[dict]:

    candidates: list[tuple[int, int, str, str]] = []
    for pat, mwe, gloss in _MWE_PATTERNS:
        for m in pat.finditer(text):
            candidates.append((m.start(), m.end(), mwe, gloss))

    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    results = []
    last_end = -1
    for start, end, mwe, gloss in candidates:
        if start >= last_end:
            results.append({"mwe": mwe, "gloss": gloss, "start": start, "end": end})
            last_end = end

    return results

def export_annotation_format_reg(episodes_raw: dict[str, str], out_filepath: str):
    """Generates the per-sentence annotation file with pattern matches."""
    with open(out_filepath, "w", encoding="utf-8") as f:
        for ep_idx, (ep_label, raw_text) in enumerate(episodes_raw.items(), 1):
            # Split raw text using same logic as in week1_analysis (with punctuation kept)
            parts_raw = re.split(r'([' + POORNA_VIRAM + DBL_DANDA + r'.?!]+)', raw_text)
            
            sents_raw = []
            for i in range(0, len(parts_raw)-1, 2):
                s = (parts_raw[i] + parts_raw[i+1]).strip()
                if s:
                    sents_raw.append(s)
            
            if len(parts_raw) % 2 == 1 and parts_raw[-1].strip():
                sents_raw.append(parts_raw[-1].strip())
                
            for sent_idx, s_raw in enumerate(sents_raw, 1):
                s_pre = preprocess(s_raw)
                s_trans = transliterate_mixed(s_pre)
                
                redup = [f"{m}-{m}" for m in find_reduplications(s_trans)]
                conj = [f"{stem}-{suf}" for stem, suf in find_conjunctive_verbs(s_trans)]
                comp = [f"{stem}+{vec}" for stem, vec in find_compound_verbs(s_trans)]
                hono = [m["form"] for m in find_honorifics(s_trans)]
                acro = find_acronyms(s_pre)
                mwe = [m["mwe"] for m in find_mwes(s_trans)]
                hyphen = [f"{l}-{r}" for l, r in find_hyphenated_pairs(s_pre)]
                
                def fmt(name, lst):
                    if not lst:
                        return f"{name}: 0"
                    return f"{name}: {len(lst)} ({', '.join(lst)})"
                
                f.write(f"T{ep_idx}_{sent_idx:03d}\n")
                # Remove newlines inside the sentence to keep it on one line
                clean_raw = " ".join(s_raw.split())
                f.write(f"{clean_raw}\n")
                
                parts = [
                    fmt("Reduplication", redup),
                    fmt("Conjunctive Verb", conj),
                    fmt("Compound Verb", comp),
                    fmt("Honorific", hono),
                    fmt("Acronym", acro),
                    fmt("MWE", mwe),
                    fmt("Hyphen Pair", hyphen)
                ]
                f.write(" | ".join(parts) + "\n\n")


# RUNNER

def run(transcript_path: str) -> dict:
    stats_path = os.path.splitext(transcript_path)[0] + "_stats_reg.txt"
    with open(stats_path, "w", encoding="utf-8") as f:

        # Load
        episodes_raw = load_transcripts(transcript_path)
        episode_order = list(episodes_raw.keys())

        results = {}

        # Week 1
        print("\n" + "="*70, file=f)
        print("WEEK 1 — CORPUS EXPLORATION & UTF-8 VALIDATION", file=f)
        print("="*70, file=f)

        w1 = week1_analysis(episodes_raw)
        results["week1"] = w1

        col_w = 22
        header = f"{'Metric':<{col_w}}" + "".join(f"{ep:>16}" for ep in episode_order) + f"{'TOTAL':>16}"
        print(header, file=f)
        print("-" * len(header), file=f)

        metrics_labels = {
            "total_chars":         "Total chars",
            "devanagari_chars":    "Devanagari chars",
            "latin_chars":         "Latin alpha chars",
            "digit_chars":         "Digit chars",
            "invalid_unicode":     "Invalid Unicode",
            "total_tokens":        "Total tokens",
            "devanagari_tokens":   "Devanagari tokens",
            "latin_tokens":        "Latin tokens",
            "sentence_count":      "Sentences (approx.)",
            "poorna_viram_count":  "Poorna viram (।)",
            "dbl_danda_count":     "Double danda (॥)",
        }

        for key, label in metrics_labels.items():
            row = f"{label:<{col_w}}"
            for ep in episode_order:
                row += f"{w1[ep][key]:>16}"
            row += f"{w1['TOTAL'][key]:>16}"
            print(row, file=f)

        print("\n✓ No invalid Unicode code points detected." if w1["TOTAL"]["invalid_unicode"] == 0
              else f"⚠ {w1['TOTAL']['invalid_unicode']} invalid Unicode code points found.")

        # Sample sentence tokenisation
        first_ep = episode_order[0]
        print(f"\nSample sentence boundaries ({first_ep}, first 5):", file=f)
        feb_pre = preprocess(episodes_raw[first_ep])
        sents = [s.strip() for s in re.split(r'[.?!]', feb_pre) if s.strip()][:5]
        for i, s in enumerate(sents, 1):
            print(f"  [{i}] {s[:80]}{'…' if len(s) > 80 else ''}", file=f)


        # Week 2
        print("\n" + "="*70, file=f)
        print("WEEK 2 — PREPROCESSING & TRANSLITERATION", file=f)
        print("="*70, file=f)

        episodes_translit = week2_pipeline(episodes_raw)
        results["week2"] = {"transliterated": episodes_translit}

        # Validation: sample sentences from each episode
        print("\nTransliteration sample (10 sentences per episode):\n", file=f)
        for ep in episode_order:
            print(f"── {ep} ──", file=f)
            sents = [s.strip() for s in re.split(r'[.?!]', episodes_translit[ep]) if s.strip()]
            for s in sents[:10]:
                if len(s) > 5:
                    print(f"  {s[:100]}{'…' if len(s) > 100 else ''}", file=f)
            print(file=f)

        # Transliteration accuracy spot-check on known Devanagari → ISO 15919 pairs
        print("Spot-check transliteration accuracy:", file=f)
        spot_checks = [
            ("नमस्कार", "namaskār"),
            ("भारत",   "bhārat"),
            ("किसान",  "kisān"),
            ("धीरे",   "dhīrē"),
            ("सरकार",  "sarkār"),
        ]
        for dev, expected in spot_checks:
            got = unicode_converter(dev)
            mark = "✓" if got == expected else "✗"
            print(f"  {mark}  {dev} → {got}  (expected {expected})", file=f)

        dev_token_count = sum(len(episodes_raw[ep].split()) for ep in episode_order)
        print(f"\n  Transliteration note: schwa deletion is word-final only.", file=f)
        print(f"  Medial schwas in ~{dev_token_count} tokens remain undeleted (known limitation).", file=f)
        print(f"  This affects compound verb stems most — e.g. sarakār vs sarkār.", file=f)

        # Week 3
        print("\n" + "="*70, file=f)
        print("WEEK 3 — PATTERN EXTRACTION (REDUPLICATION, ACRONYMS, HYPHENATED PAIRS)", file=f)
        print("="*70, file=f)

        w3_results = {}

        for pattern_name, extract_fn, apply_on in [
            ("Reduplication",    find_reduplications,    "transliterated"),
            ("Acronyms",         find_acronyms,          "raw"),
            ("Hyphenated Pairs", find_hyphenated_pairs,  "raw"),
        ]:
            print(f"\n── {pattern_name} ──", file=f)
            pat_results = {}

            for ep in episode_order:
                text = (episodes_translit[ep] if apply_on == "transliterated"
                        else preprocess(episodes_raw[ep]))
                matches = extract_fn(text)

                if pattern_name == "Reduplication":
                    # matches are captured groups (just the word); build full form
                    full_matches = [f"{m}-{m}" for m in matches]
                    unique = sorted(set(full_matches))
                    freq   = {m: full_matches.count(m) for m in unique}
                elif pattern_name == "Acronyms":
                    unique = sorted(set(matches))
                    freq   = {m: matches.count(m) for m in unique}
                else:  # Hyphenated pairs
                    full_matches = [f"{l}-{r}" for l, r in matches]
                    unique = sorted(set(full_matches))
                    freq   = {m: full_matches.count(m) for m in unique}

                pat_results[ep] = {
                    "total_instances": len(matches),
                    "unique_types":    len(unique),
                    "frequency":       freq,
                    "top10":           sorted(freq.items(), key=lambda x: -x[1])[:10],
                }

            w3_results[pattern_name] = pat_results

            # Print summary table
            print(f"  {'Episode':<20} {'Total instances':>18} {'Unique types':>14}", file=f)
            print("  " + "-"*54, file=f)
            for ep in episode_order:
                r = pat_results[ep]
                print(f"  {ep:<20} {r['total_instances']:>18} {r['unique_types']:>14}", file=f)

            # Print top examples
            print(f"\n  Top examples (across all episodes):", file=f)
            all_freqs: dict[str, int] = {}
            for ep in episode_order:
                for item, cnt in pat_results[ep]["frequency"].items():
                    all_freqs[item] = all_freqs.get(item, 0) + cnt
            for item, cnt in sorted(all_freqs.items(), key=lambda x: -x[1])[:15]:
                print(f"    {item}  ({cnt}×)", file=f)

        results["week3"] = w3_results


        # Week 4
        print("\n" + "="*70, file=f)
        print("WEEK 4 — PATTERN EXTRACTION (CONJ. VERBS, COMPOUND VERBS, HONORIFICS, MWEs)", file=f)
        print("="*70, file=f)

        w4_results = {}

        # 4a. Conjunctive Verbs
        print("\n── Conjunctive Verbs (-kar / -kē) ──", file=f)
        cv_results = {}
        for ep in episode_order:
            text = episodes_translit[ep]
            matches = find_conjunctive_verbs(text)
            full = [f"{stem}-{suf}" for stem, suf in matches]
            unique = sorted(set(full))
            freq = {m: full.count(m) for m in unique}
            cv_results[ep] = {
                "total_instances": len(matches),
                "unique_types":    len(unique),
                "frequency":       freq,
                "top10":           sorted(freq.items(), key=lambda x: -x[1])[:10],
            }
        w4_results["Conjunctive Verbs"] = cv_results

        print(f"  {'Episode':<20} {'Total instances':>18} {'Unique types':>14}", file=f)
        print("  " + "-"*54, file=f)
        for ep in episode_order:
            r = cv_results[ep]
            print(f"  {ep:<20} {r['total_instances']:>18} {r['unique_types']:>14}", file=f)
        all_cv: dict[str, int] = {}
        for ep in episode_order:
            for item, cnt in cv_results[ep]["frequency"].items():
                all_cv[item] = all_cv.get(item, 0) + cnt
        print("\n  Top examples:", file=f)
        for item, cnt in sorted(all_cv.items(), key=lambda x: -x[1])[:15]:
            print(f"    {item}  ({cnt}×)", file=f)

        # 4b. Compound Verbs
        print("\n── Compound Verbs (stem + vector) ──", file=f)
        compv_results = {}
        for ep in episode_order:
            text = episodes_translit[ep]
            matches = find_compound_verbs(text)
            full = [f"{stem}+{vec}" for stem, vec in matches]
            unique = sorted(set(full))
            freq = {m: full.count(m) for m in unique}
            compv_results[ep] = {
                "total_instances": len(matches),
                "unique_types":    len(unique),
                "frequency":       freq,
                "top10":           sorted(freq.items(), key=lambda x: -x[1])[:10],
            }
        w4_results["Compound Verbs"] = compv_results

        print(f"  {'Episode':<20} {'Total instances':>18} {'Unique types':>14}", file=f)
        print("  " + "-"*54, file=f)
        for ep in episode_order:
            r = compv_results[ep]
            print(f"  {ep:<20} {r['total_instances']:>18} {r['unique_types']:>14}", file=f)
        all_compv: dict[str, int] = {}
        for ep in episode_order:
            for item, cnt in compv_results[ep]["frequency"].items():
                all_compv[item] = all_compv.get(item, 0) + cnt
        print("\n  Top examples (stem+vector):", file=f)
        for item, cnt in sorted(all_compv.items(), key=lambda x: -x[1])[:15]:
            print(f"    {item}  ({cnt}×)", file=f)

        # 4c. Honorifics
        print("\n── Honorifics (āp / tum / tū) ──", file=f)
        hon_results = {}
        for ep in episode_order:
            text = episodes_translit[ep]
            matches = find_honorifics(text)
            by_register: dict[str, int] = {}
            for m in matches:
                key = f"{m['form']} [{m['register']}]"
                by_register[key] = by_register.get(key, 0) + 1
            hon_results[ep] = {
                "total_instances": len(matches),
                "unique_types":    len(by_register),
                "frequency":       by_register,
            }
        w4_results["Honorifics"] = hon_results

        print(f"  {'Episode':<20} {'Total instances':>18} {'Unique types':>14}", file=f)
        print("  " + "-"*54, file=f)
        for ep in episode_order:
            r = hon_results[ep]
            print(f"  {ep:<20} {r['total_instances']:>18} {r['unique_types']:>14}", file=f)
        all_hon: dict[str, int] = {}
        for ep in episode_order:
            for item, cnt in hon_results[ep]["frequency"].items():
                all_hon[item] = all_hon.get(item, 0) + cnt
        print("\n  Register breakdown (all episodes):", file=f)
        for item, cnt in sorted(all_hon.items(), key=lambda x: -x[1]):
            print(f"    {item}  ({cnt}×)", file=f)

        # 4d. Multi-Word Expressions
        print("\n── Multi-Word Expressions (lexicon-driven) ──", file=f)
        mwe_results = {}
        for ep in episode_order:
            text = episodes_translit[ep]
            matches = find_mwes(text)
            freq: dict[str, int] = {}
            for m in matches:
                key = f"{m['mwe']}  [{m['gloss']}]"
                freq[key] = freq.get(key, 0) + 1
            mwe_results[ep] = {
                "total_instances": len(matches),
                "unique_types":    len(freq),
                "frequency":       freq,
                "top10":           sorted(freq.items(), key=lambda x: -x[1])[:10],
            }
        w4_results["MWEs"] = mwe_results

        print(f"  {'Episode':<20} {'Total instances':>18} {'Unique types':>14}", file=f)
        print("  " + "-"*54, file=f)
        for ep in episode_order:
            r = mwe_results[ep]
            print(f"  {ep:<20} {r['total_instances']:>18} {r['unique_types']:>14}", file=f)
        all_mwe: dict[str, int] = {}
        for ep in episode_order:
            for item, cnt in mwe_results[ep]["frequency"].items():
                all_mwe[item] = all_mwe.get(item, 0) + cnt
        print("\n  MWE types found (all episodes):", file=f)
        for item, cnt in sorted(all_mwe.items(), key=lambda x: -x[1]):
            print(f"    {item}  ({cnt}×)", file=f)

        results["week4"] = w4_results

        # Generate the requested annotation file
        annotation_path = os.path.splitext(transcript_path)[0] + "_annotation_reg.txt"
        try:
            export_annotation_format_reg(episodes_raw, annotation_path)
            print(f"\n  Exported sentence-level annotation to: {annotation_path}", file=f)
        except Exception as e:
            print(f"\n  Error exporting annotation: {e}", file=f)

        print("\n" + "="*70, file=f)
        print("PIPELINE COMPLETE", file=f)
        print("="*70, file=f)

    return results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else print("Please provide a path to the transcript file")
    run(path)

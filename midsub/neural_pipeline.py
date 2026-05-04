import re
import sys
import os
import time
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import load_transcripts, preprocess, week2_pipeline

# STANZA INITIALISATION

def load_stanza_pipeline():
    try:
        import stanza
    except ImportError:
        raise ImportError(
            "Stanza is not installed.  Run:\n"
            "    pip install stanza\n"
            "    python -c \"import stanza; stanza.download('hi')\""
        )
    stanza.download("hi", verbose=False)

    nlp = stanza.Pipeline(
        lang="hi",
        processors="tokenize,pos,lemma,depparse",
        verbose=False,
        use_gpu=False,          # set True if a CUDA GPU is available
        tokenize_no_ssplit=False,  # let Stanza split sentences
    )
    return nlp


# VECTOR VERB LEMMAS  

VECTOR_LEMMAS = {
    "लेना",   # lēnā
    "देना",   # dēnā
    "जाना",   # jānā
    "आना",    # ānā
    "पड़ना",  # paṛnā
}

# ISO 15919 forms (for transliterated text fallback)
VECTOR_ISO = {"lēnā", "dēnā", "jānā", "ānā", "paṛnā"}

# Honorific pronoun lemmas
HONORIFIC_LEMMAS = {
    "आप":  "formal",
    "तुम": "familiar",
    "तू":  "intimate",
}

# MWE postposition lemmas to watch for
MWE_ADP_LEMMAS = {
    "बाद", "आगे", "बीच", "साथ", "लिए", "बारे",
    "तहत", "वजह", "तरफ", "जगह", "रूप", "द्वारा",
    "अनुसार", "अलावा",
}

# Acronym pattern (identical to regex rule)
ACRONYM_RE = re.compile(r'(?<!-)\b[A-Z]{2,}\b')

# Reduplication suffix pattern
CONJ_VERB_SURFACE_RE = re.compile(r'\b\w{3,}-?(kar|ke|kē)\b', re.IGNORECASE)

# PER-SENTENCE EXTRACTION FUNCTIONS

def extract_from_sentence(sent) -> dict:
    words = sent.words
    results = defaultdict(list)

    for i, w in enumerate(words):
        text   = w.text   or ""
        lemma  = w.lemma  or ""
        upos   = w.upos   or ""

        # Reduplication
        if "-" in text:
            halves = text.split("-", 1)
            if len(halves) == 2 and halves[0].lower() == halves[1].lower():
                results["Reduplication"].append(text)

        prev_text = words[i-1].text or ""
        prev_upos = words[i-1].upos or ""
        if (i > 0
                and lemma and len(lemma) > 1
                and lemma == (words[i-1].lemma or "")
                and len(prev_text.strip()) > 1
                and prev_upos not in ("PUNCT", "SYM", "X", "NUM")):
            results["Reduplication"].append(f"{prev_text}-{text}")

        # Conjunctive Verbs
        if upos == "VERB" and re.search(r'-?(kar|ke|kē)$', text, re.IGNORECASE):
            results["Conjunctive Verbs"].append(text)

        # Compound Verbs
        if upos == "VERB":
            for j in range(i + 1, min(i + 4, len(words))):
                nw = words[j]
                if nw.upos == "AUX" and (nw.lemma in VECTOR_LEMMAS):
                    results["Compound Verbs"].append(f"{text}+{nw.text}")
                    break

        # Honorifics
        if upos == "PRON" and lemma in HONORIFIC_LEMMAS:
            register = HONORIFIC_LEMMAS[lemma]
            results["Honorifics"].append(f"{text} [{register}]")

        # Acronyms
        if ACRONYM_RE.match(text):
            results["Acronyms"].append(text)

        # MWEs
        if upos == "ADP" and lemma in MWE_ADP_LEMMAS:
            head_idx = (w.head or 1) - 1   # head is 1-indexed
            head_text = words[head_idx].text if 0 <= head_idx < len(words) else ""
            results["MWEs"].append(f"{head_text} {text}")

        # Hyphenated Pairs
        if "-" in text:
            halves = text.split("-", 1)
            if (len(halves) == 2
                    and len(halves[0]) >= 2
                    and len(halves[1]) >= 2
                    and halves[0].lower() != halves[1].lower()):
                results["Hyphenated Pairs"].append(text)

    return results


# EPISODE-LEVEL PROCESSING

def process_episode(nlp, raw_text: str, label: str) -> tuple[dict, dict, list]:
    preprocessed = preprocess(raw_text)

    t0 = time.time()
    doc = nlp(preprocessed)
    elapsed = time.time() - t0

    n_sents = len(doc.sentences)
    speed   = n_sents / elapsed if elapsed > 0 else float("inf")

    print(f"  [{label}] {n_sents} sentences processed in {elapsed:.1f}s "
          f"({speed:.0f} sent/s)")

    all_matches: dict[str, list] = defaultdict(list)
    sentence_annotations = []
    
    for sent in doc.sentences:
        per_sent = extract_from_sentence(sent)
        for pattern, matches in per_sent.items():
            all_matches[pattern].extend(matches)
            
        def fmt(name, key):
            lst = per_sent.get(key, [])
            if not lst:
                return f"{name}: 0"
            return f"{name}: {len(lst)} ({', '.join(lst)})"
            
        clean_raw = " ".join((sent.text or "").split())
        parts = [
            fmt("Reduplication", "Reduplication"),
            fmt("Conjunctive Verb", "Conjunctive Verbs"),
            fmt("Compound Verb", "Compound Verbs"),
            fmt("Honorific", "Honorifics"),
            fmt("Acronym", "Acronyms"),
            fmt("MWE", "MWEs"),
            fmt("Hyphen Pair", "Hyphenated Pairs")
        ]
        sentence_annotations.append((clean_raw, " | ".join(parts)))

    episode_result = {}
    PATTERNS = [
        "Reduplication", "Conjunctive Verbs", "Compound Verbs",
        "Honorifics", "Acronyms", "MWEs", "Hyphenated Pairs",
    ]
    for pattern in PATTERNS:
        items = all_matches.get(pattern, [])
        freq: dict[str, int] = {}
        for item in items:
            freq[item] = freq.get(item, 0) + 1
        episode_result[pattern] = {
            "total":  len(items),
            "unique": len(freq),
            "freq":   freq,
            "top10":  sorted(freq.items(), key=lambda x: -x[1])[:10],
        }

    return episode_result, {"sentences": n_sents, "time_s": round(elapsed, 2),
                             "speed_sent_per_s": round(speed, 1)}, sentence_annotations


# EVALUATION AGAINST GOLD STANDARD

def evaluate_against_gold(
    neural_results: dict,
    gold_path: str = "gold_standard.json"
) -> dict | None:
    if not os.path.exists(gold_path):
        return None

    with open(gold_path, encoding="utf-8") as fh:
        gold = json.load(fh)

    eval_results = {}
    for ep, patterns in neural_results.items():
        if ep not in gold:
            continue
        ep_eval = {}
        for pat, data in patterns.items():
            if pat not in gold[ep]:
                continue
            pred_set = set(data["freq"].keys())
            gold_set = set(gold[ep][pat])
            tp = len(pred_set & gold_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            ep_eval[pat] = {
                "precision": round(precision, 3),
                "recall":    round(recall, 3),
                "f1":        round(f1, 3),
                "tp": tp, "fp": fp, "fn": fn,
            }
        eval_results[ep] = ep_eval

    return eval_results


def export_annotation_format_neural(episode_order: list[str], all_sentence_annotations: dict, out_filepath: str):
    """Generates the per-sentence annotation file with pattern matches for neural pipeline."""
    with open(out_filepath, "w", encoding="utf-8") as f:
        for ep_idx, ep in enumerate(episode_order, 1):
            sent_anns = all_sentence_annotations[ep]
            for sent_idx, (clean_raw, parts_str) in enumerate(sent_anns, 1):
                f.write(f"T{ep_idx}_{sent_idx:03d}\n")
                f.write(f"{clean_raw}\n")
                f.write(f"{parts_str}\n\n")

# MAIN RUNNER

def run(transcript_path: str) -> dict:
    stats_path = os.path.splitext(transcript_path)[0] + "_stats_neural.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        print("\n" + "="*70, file=f)
        print("WEEK 4 (TASK 2) – NEURAL COMPARATIVE PIPELINE", file=f)
        print("="*70, file=f)

        # Load corpus
        episodes_raw   = load_transcripts(transcript_path)
        episode_order  = list(episodes_raw.keys())

        # Load Stanza
        print("\nLoading Stanza Hindi pipeline …", file=f)
        nlp = load_stanza_pipeline()
        print("Stanza pipeline ready.\n", file=f)

        PATTERNS = [
            "Reduplication", "Conjunctive Verbs", "Compound Verbs",
            "Honorifics", "Acronyms", "MWEs", "Hyphenated Pairs",
        ]

        neural_results = {}
        speed_log      = {}
        all_sentence_annotations = {}

        print("Processing episodes:", file=f)
        for ep in episode_order:
            ep_result, timing, sent_anns = process_episode(nlp, episodes_raw[ep], ep)
            neural_results[ep] = ep_result
            speed_log[ep]      = timing
            all_sentence_annotations[ep] = sent_anns

        print("\n" + "="*70, file=f)
        print("RESULTS BY PATTERN", file=f)
        print("="*70, file=f)

        for pat in PATTERNS:
            print(f"\n── {pat} ──", file=f)
            print(f"  {'Episode':<20} {'Total':>10} {'Unique':>10}", file=f)
            print("  " + "-"*42, file=f)
            for ep in episode_order:
                r = neural_results[ep][pat]
                print(f"  {ep:<20} {r['total']:>10} {r['unique']:>10}", file=f)

            # Aggregate top examples
            agg: dict[str, int] = {}
            for ep in episode_order:
                for item, cnt in neural_results[ep][pat]["freq"].items():
                    agg[item] = agg.get(item, 0) + cnt
            print(f"\n  Top examples (all episodes):", file=f)
            for item, cnt in sorted(agg.items(), key=lambda x: -x[1])[:15]:
                print(f"    {item}  ({cnt}×)", file=f)

        print("\n" + "="*70, file=f)
        print("SPEED SUMMARY", file=f)
        print("="*70, file=f)
        print(f"  {'Episode':<20} {'Sentences':>12} {'Time (s)':>12} {'Sent/s':>10}", file=f)
        print("  " + "-"*56, file=f)
        for ep in episode_order:
            t = speed_log[ep]
            print(f"  {ep:<20} {t['sentences']:>12} {t['time_s']:>12.2f} "
                  f"{t['speed_sent_per_s']:>10.1f}")

        print(
            "\n  NOTE: Compare Stanza sent/s against the regex pipeline's throughput\n"
            "  (measured separately in pipeline.py).  The regex approach typically\n"
            "  runs 50-200× faster; the neural model trades speed for morphological\n"
            "  awareness (lemmatisation, POS, dependency structure)."
        )

        # Gold-standard evaluation
        eval_results = evaluate_against_gold(neural_results)
        if eval_results:
            print("\n" + "="*70, file=f)
            print("EVALUATION AGAINST GOLD STANDARD", file=f)
            print("="*70, file=f)
            for ep, pats in eval_results.items():
                print(f"\n  {ep}", file=f)
                print(f"    {'Pattern':<22} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}", file=f)
                print("    " + "-"*56, file=f)
                for pat, scores in pats.items():
                    print(f"    {pat:<22} {scores['precision']:>6.3f} {scores['recall']:>6.3f} "
                          f"{scores['f1']:>6.3f} {scores['tp']:>5} {scores['fp']:>5} {scores['fn']:>5}")
        else:
            print(
                "\n  No gold_standard.json found.  To enable automatic P/R/F1 evaluation,\n"
                "  create a gold_standard.json file in the same directory as this script.\n"
                "  See the module docstring for the required format."
            )

        # Generate the requested annotation file
        annotation_path = os.path.splitext(transcript_path)[0] + "_annotation_neural.txt"
        try:
            export_annotation_format_neural(episode_order, all_sentence_annotations, annotation_path)
            print(f"\n  Exported sentence-level annotation to: {annotation_path}", file=f)
        except Exception as e:
            print(f"\n  Error exporting annotation: {e}", file=f)

        print("\n" + "="*70, file=f)
        print("NEURAL PIPELINE COMPLETE", file=f)
        print("="*70, file=f)

    return neural_results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else print("Please provide a path to the transcript file")
    run(path)

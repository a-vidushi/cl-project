import re
import sys
from collections import defaultdict

CATEGORIES = [
    "Reduplication",
    "Conjunctive Verb",
    "Compound Verb",
    "Honorific",
    "Acronym",
    "MWE",
    "Hyphen Pair",
]

def parse_file(filepath):
    records = []
    with open(filepath, encoding="utf-8-sig") as f:
        content = f.read()

    blocks = re.split(r'\n(?=T\d+_\d+)', content.strip())

    for block in blocks:
        lines = [l for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue

        sent_id = lines[0].strip()

        ann_line = next(
            (l for l in reversed(lines) if re.search(r'Reduplication\s*:', l)),
            None
        )
        if ann_line is None:
            continue

        sentence = " ".join(
            l for l in lines[1:] if l != ann_line
        ).strip()

        sentence_norm = normalise(sentence)

        cats = {}
        for cat in CATEGORIES:
            pat = rf'{re.escape(cat)}\s*:\s*(\d+)'
            m = re.search(pat, ann_line)
            cats[cat] = 1 if (m and int(m.group(1)) > 0) else 0

        records.append((sent_id, sentence_norm, cats))

    return records


def normalise(text):
    text = text.strip().lower()

    # remove tags in parentheses
    text = re.sub(r'\(.*?\)', '', text)

    # whitespace
    text = re.sub(r'\s+', ' ', text)

    # punctuation normalization
    text = text.replace('।', '.')
    text = text.replace('॥', '.')
    text = text.replace('–', '-').replace('—', '-')

    # quotes
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")

    # remove trailing punctuation
    text = re.sub(r'[.?!]+$', '', text)

    return text.strip()

def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def evaluate(gold_path, pred_path):
    gold_records = parse_file(gold_path)
    pred_records = parse_file(pred_path)

    print(f"Sentences in gold          : {len(gold_records)}")
    print(f"Sentences in other file    : {len(pred_records)}")

    # --- FIX: exact sentence matching with one-to-one mapping ---
    pred_map = {}
    for p_id, p_sent, p_cats in pred_records:
        pred_map.setdefault(p_sent, []).append((p_id, p_cats))

    matched_pairs = []
    unmatched_gold = []
    unmatched_pred_ids = set(p_id for p_id, _, _ in pred_records)

    for g_id, g_sent, g_cats in gold_records:
        if g_sent in pred_map and pred_map[g_sent]:
            p_id, p_cats = pred_map[g_sent].pop(0)  # one-to-one match
            matched_pairs.append((g_id, g_sent, g_cats, p_id, g_sent, p_cats))
            unmatched_pred_ids.discard(p_id)
        else:
            unmatched_gold.append(g_id)

    print(f"Matched (evaluated)        : {len(matched_pairs)}")
    print(f"Unmatched gold (skipped)   : {len(unmatched_gold)}")
    print(f"Unmatched pred (skipped)   : {len(unmatched_pred_ids)}")

    if unmatched_gold:
        print(f"  Gold-only IDs: {unmatched_gold[:10]}{'...' if len(unmatched_gold) > 10 else ''}")
    if unmatched_pred_ids:
        print(f"  Pred-only IDs: {list(unmatched_pred_ids)[:10]}{'...' if len(unmatched_pred_ids) > 10 else ''}")

    print()

    if not matched_pairs:
        print("No matches found. Check normalization.")
        sys.exit(1)

    total_sentences = len(matched_pairs)

    cat_tp      = defaultdict(int)
    cat_fp      = defaultdict(int)
    cat_fn      = defaultdict(int)
    cat_correct = defaultdict(int)
    fully_correct = 0

    for g_id, g_sent, g_cats, p_id, p_sent, p_cats in matched_pairs:
        sentence_fully_correct = True

        for cat in CATEGORIES:
            gv = g_cats.get(cat, 0)
            pv = p_cats.get(cat, 0)

            cat_tp[cat]      += int(gv == 1 and pv == 1)
            cat_fp[cat]      += int(gv == 0 and pv == 1)
            cat_fn[cat]      += int(gv == 1 and pv == 0)
            cat_correct[cat] += int(gv == pv)

            if gv != pv:
                sentence_fully_correct = False

        fully_correct += int(sentence_fully_correct)

    col = f"{'Category':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}  TP   FP   FN"
    print(col)
    print("-" * len(col))

    total_tp = total_fp = total_fn = total_correct = 0

    for cat in CATEGORIES:
        tp, fp, fn = cat_tp[cat], cat_fp[cat], cat_fn[cat]
        p, r, f    = prf(tp, fp, fn)
        acc        = cat_correct[cat] / total_sentences

        total_tp      += tp
        total_fp      += fp
        total_fn      += fn
        total_correct += cat_correct[cat]

        print(f"{cat:<20} {acc:>10.4f} {p:>10.4f} {r:>10.4f} {f:>10.4f}  {tp:>3}  {fp:>3}  {fn:>3}")

    print("-" * len(col))

    micro_p, micro_r, micro_f = prf(total_tp, total_fp, total_fn)
    micro_acc  = total_correct / (total_sentences * len(CATEGORIES))
    sent_acc   = fully_correct / total_sentences

    print(f"{'OVERALL (micro)':<20} {micro_acc:>10.4f} {micro_p:>10.4f} {micro_r:>10.4f} {micro_f:>10.4f}  "
          f"{total_tp:>3}  {total_fp:>3}  {total_fn:>3}")
    print()
    print(f"Sentence-level accuracy (all categories correct): "
          f"{sent_acc:.4f}  ({fully_correct}/{total_sentences})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <gold_file> <pred_file>")
        sys.exit(1)

    evaluate(sys.argv[1], sys.argv[2])
# Hindi NLP Pipeline — Final Submission

## Quick Start — Run in Order

```bash
# 1. Regex pipeline (transcripts)
python pipeline.py transcripts_raw.txt                # or: uv run pipeline.py transcripts_raw.txt

# 2. Neural pipeline (transcripts) — requires Stanza
python neural_pipeline.py transcripts_raw.txt         # or: uv run neural_pipeline.py transcripts_raw.txt

# 3. Evaluate regex vs gold
python metrics.py transcripts_raw_annotation_gold.txt transcripts_raw_annotation_reg.txt
# or: uv run metrics.py transcripts_raw_annotation_gold.txt transcripts_raw_annotation_reg.txt

# 4. Evaluate neural vs gold
python metrics.py transcripts_raw_annotation_gold.txt transcripts_raw_annotation_neural.txt
# or: uv run metrics.py transcripts_raw_annotation_gold.txt transcripts_raw_annotation_neural.txt

# 5. Repeat for social media corpus
python pipeline.py social/comments_devanagari.txt     # or: uv run pipeline.py social/comments_devanagari.txt
python neural_pipeline.py social/comments_devanagari.txt
# or: uv run neural_pipeline.py social/comments_devanagari.txt
python metrics.py social/comments_devanagari_annotation_gold.txt social/comments_devanagari_annotation_reg.txt
# or: uv run metrics.py social/comments_devanagari_annotation_gold.txt social/comments_devanagari_annotation_reg.txt
python metrics.py social/comments_devanagari_annotation_gold.txt social/comments_devanagari_annotation_neural.txt
# or: uv run metrics.py social/comments_devanagari_annotation_gold.txt social/comments_devanagari_annotation_neural.txt
```

---

A computational linguistics project that analyses Hindi-language text using two parallel pipelines — a **rule-based regex pipeline** and a **neural Stanza-based pipeline** — to extract seven linguistic patterns from mixed Devanagari + English corpora.

Two corpora are analysed:
1. **Mann Ki Baat** radio transcripts (primary corpus)
2. **YouTube comments** scraped from Mann Ki Baat videos (social media corpus)

## Linguistic Patterns Extracted

| # | Pattern | Description |
|---|---------|-------------|
| 1 | Reduplication | Hyphen-joined word repetitions (e.g. *dhīrē-dhīrē*, *alag-alag*) |
| 2 | Acronyms | Consecutive uppercase Latin letters (e.g. AI, KYC, UPI) |
| 3 | Hyphenated Pairs | Non-reduplicative hyphenated tokens (e.g. *Start-Up*, *re-KYC*) |
| 4 | Conjunctive Verbs | Absolutive verb forms ending in *-kar / -kē* |
| 5 | Compound Verbs | Primary verb + vector/light verb (e.g. *lēnā*, *dēnā*, *jānā*) |
| 6 | Honorifics | Hindi T-V pronominal forms (*āp*, *tum*, *tū*) |
| 7 | Multi-Word Expressions | Fixed postpositional phrases (*kē sāth*, *kē liyē*, etc.) |

## Folder Structure

```
./
├── pipeline.py                                  # Rule-based regex pipeline (Weeks 1–4)
├── neural_pipeline.py                           # Stanza-based neural pipeline
├── metrics.py                                   # Evaluation script (gold vs predicted annotations)
├── unicode_converter.py                         # Devanagari → ISO 15919 transliterator
│
├── transcripts_raw.txt                          # Source corpus — Mann Ki Baat transcripts (Devanagari)
├── transcripts_raw_transliteration.txt          # Transliterated corpus (ISO 15919)
├── transcripts_raw_annotation_gold.txt          # Gold-standard per-sentence annotations
├── transcripts_raw_annotation_reg.txt           # Regex pipeline per-sentence annotations
├── transcripts_raw_annotation_neural.txt        # Neural pipeline per-sentence annotations
├── transcripts_raw_stats_reg.txt                # Regex pipeline full statistics output
├── transcripts_raw_stats_neural.txt             # Neural pipeline full statistics output
├── transcripts_raw_metrics_reg.txt              # Evaluation metrics: regex vs gold
├── transcripts_raw_metrics_neural.txt           # Evaluation metrics: neural vs gold
│
├── social/                                      # Social media corpus (YouTube comments)
│   ├── commentscraper.py                        # YouTube comment scraper (uses yt-dlp)
│   ├── normalize_comments.py                    # Comment normalization & noise filtering
│   ├── comments_devanagari.txt                  # Cleaned Devanagari-script comments
│   ├── comments_devanagari_transliteration.txt  # Transliterated comments (ISO 15919)
│   ├── comments_devanagari_annotation_gold.txt  # Gold-standard annotations for comments
│   ├── comments_devanagari_annotation_reg.txt   # Regex pipeline annotations for comments
│   ├── comments_devanagari_annotation_neural.txt# Neural pipeline annotations for comments
│   ├── comments_devanagari_stats_reg.txt        # Regex pipeline stats for comments
│   ├── comments_devanagari_stats_neural.txt     # Neural pipeline stats for comments
│   ├── comments_devanagari_metrics_reg.txt      # Evaluation metrics: regex vs gold (comments)
│   └── comments_devanagari_metrics_neural.txt   # Evaluation metrics: neural vs gold (comments)
│
└── project_report.pdf                            # Final project report
```

## Scripts

### `pipeline.py` — Rule-Based Regex Pipeline

Implements the full analysis pipeline across four weekly stages:

| Week | Stage | What it does |
|------|-------|--------------|
| 1 | Corpus exploration & UTF-8 validation | Character/token statistics, sentence boundary detection |
| 2 | Preprocessing & transliteration | NFC normalisation, punctuation normalisation, Devanagari → ISO 15919 |
| 3 | Pattern extraction (Part 1) | Reduplication, acronyms, hyphenated pairs |
| 4 | Pattern extraction (Part 2) | Conjunctive verbs, compound verbs, honorifics, MWEs |

Produces a `*_stats_reg.txt` statistics file and a `*_annotation_reg.txt` per-sentence annotation file.

```bash
python pipeline.py transcripts_raw.txt
```

### `neural_pipeline.py` — Stanza Neural Pipeline

Uses Stanza's Hindi model (tokenizer, POS tagger, lemmatiser, dependency parser) to extract the same seven patterns with morphological awareness. Imports preprocessing utilities from `pipeline.py`.

Produces a `*_stats_neural.txt` statistics file and a `*_annotation_neural.txt` per-sentence annotation file.

```bash
python neural_pipeline.py transcripts_raw.txt
```

> **Note:** The neural pipeline is significantly slower than the regex pipeline (~50–200× fewer sentences per second) but provides morphological awareness through lemmatisation, POS tags, and dependency structure.

### `metrics.py` — Evaluation Against Gold Standard

Compares a predicted annotation file (regex or neural) against a gold-standard annotation file. Reports per-category accuracy, precision, recall, F1, and sentence-level accuracy.

```bash
python metrics.py transcripts_raw_annotation_gold.txt transcripts_raw_annotation_reg.txt
python metrics.py transcripts_raw_annotation_gold.txt transcripts_raw_annotation_neural.txt
```

### `unicode_converter.py` — Devanagari → ISO 15919 Transliterator

Converts Devanagari Unicode text to ISO 15919 romanisation. Handles consonant clusters, matras, nukta forms, and word-final schwa deletion. Used internally by both pipelines.

### `social/commentscraper.py` — YouTube Comment Scraper

Scrapes comments from specified YouTube video IDs using `yt-dlp`. Outputs CSV and plain-text files with script-detection summaries.

```bash
python social/commentscraper.py
```

### `social/normalize_comments.py` — Comment Normalisation

Cleans and filters raw YouTube comments: deduplication, spam/noise removal, script detection (Devanagari / Roman / mixed), NFC normalisation, emoji removal, and repeated character/word collapsing. Outputs per-script plain-text files ready for pipeline processing.

```bash
python social/normalize_comments.py -i social/comments_raw.txt -o social/normalized/
```

## Output Files

For each corpus (transcripts and social comments), the pipelines produce a parallel set of outputs:

| File suffix | Description |
|-------------|-------------|
| `_transliteration.txt` | Full transliterated text (Devanagari → ISO 15919) |
| `_annotation_gold.txt` | Manually annotated gold standard (per-sentence pattern labels) |
| `_annotation_reg.txt` | Regex pipeline's per-sentence annotation output |
| `_annotation_neural.txt` | Neural pipeline's per-sentence annotation output |
| `_stats_reg.txt` | Regex pipeline statistics (corpus exploration, transliteration samples, pattern counts) |
| `_stats_neural.txt` | Neural pipeline statistics (pattern counts, speed summary) |
| `_metrics_reg.txt` | Evaluation results comparing regex annotations against gold |
| `_metrics_neural.txt` | Evaluation results comparing neural annotations against gold |

## Setup

### Install Stanza Hindi model

```bash
python -c "import stanza; stanza.download('hi')"
```

### Install yt-dlp (for comment scraping)

```bash
pip install yt-dlp
```

## Known Limitations

- **Schwa deletion** is word-final only; medial schwas remain undeleted (e.g. *sarakār* instead of *sarkār*).
- The **compound verb detector** may produce false positives when a non-verb token happens to precede a vector verb within the matching window.
- **Stanza's Hindi lemmatiser** is unreliable for code-switched English tokens.

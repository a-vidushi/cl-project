# Hindi NLP Pipeline — *Mann Ki Baat* Corpus Analysis

A computational linguistics project that analyses Hindi-language transcripts from the *Mann Ki Baat* radio programme. The project implements two parallel pipelines — a **rule-based regex pipeline** and a **neural Stanza-based pipeline** — to extract seven linguistic patterns from a mixed Devanagari + English corpus.

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

## Project Structure

```
proj/
├── data/
│   ├── raw/                  # Source corpus (Audio Transcripts.docx)
│   └── processed/            # Extracted plain-text file (transcripts_raw.txt)
├── work/
│   ├── pipeline.py           # Rule-based regex pipeline (Weeks 1–4)
│   ├── neural_pipeline.py    # Stanza-based neural pipeline (Week 4, Task 2)
│   └── unicode_converter.py  # Devanagari → ISO 15919 transliterator
├── outputs/
│   ├── pipeline_run.txt      # Sample output from the regex pipeline
│   └── neural_run.txt        # Sample output from the neural pipeline
├── docs/                     # Progress reports & proposal
├── samples/                  # Project outline & reference PDFs
├── pyproject.toml            # Project metadata & dependencies
└── uv.lock                   # Dependency lock file
```


## Setup

### 1. Download the Stanza Hindi model

```bash
uv run python -c "import stanza; stanza.download('hi')"
```

Or with pip:

```bash
python -c "import stanza; stanza.download('hi')"
```

## Running the Pipelines

### Rule-Based Regex Pipeline

This runs all four weeks of analysis (corpus exploration, preprocessing & transliteration, and regex-based pattern extraction):

```bash
uv run python work/pipeline.py data/processed/transcripts_raw.txt
```

Or without uv:

```bash
python work/pipeline.py data/processed/transcripts_raw.txt
```

**What it does (by week):**

| Week | Stage | Output |
|------|-------|--------|
| 1 | Corpus exploration & UTF-8 validation | Character/token statistics, sentence boundary detection |
| 2 | Preprocessing & transliteration | NFC normalisation, punctuation normalisation, Devanagari → ISO 15919 |
| 3 | Pattern extraction (Part 1) | Reduplication, acronyms, hyphenated pairs |
| 4 | Pattern extraction (Part 2) | Conjunctive verbs, compound verbs, honorifics, MWEs |

### Neural (Stanza) Pipeline

This runs the comparative neural pipeline using Stanza's Hindi model for POS tagging, lemmatisation, and dependency parsing:

```bash
uv run python work/neural_pipeline.py data/processed/transcripts_raw.txt
```

Or without uv:

```bash
python work/neural_pipeline.py data/processed/transcripts_raw.txt
```

> **Note:** The neural pipeline is significantly slower than the regex pipeline (Stanza processes ~50–200× fewer sentences per second) but provides morphological awareness through lemmatisation, POS tags, and dependency structure.

### Saving Output to a File

To save the output for review:

```bash
uv run python work/pipeline.py data/processed/transcripts_raw.txt > outputs/pipeline_run.txt
uv run python work/neural_pipeline.py data/processed/transcripts_raw.txt > outputs/neural_run.txt
```

## Running Tests

```bash
uv run pytest
```

## Known Limitations

- **Schwa deletion** is word-final only; medial schwas remain undeleted (e.g. *sarakār* instead of *sarkār*).
- The **compound verb detector** may produce false positives when a non-verb token happens to precede a vector verb within the matching window.
- **Stanza's Hindi lemmatiser** is unreliable for code-switched English tokens.

# Redrob Talent Intelligence Engine

> AI-powered candidate ranking system that goes beyond keyword matching to rank candidates for a Senior AI Engineer role using semantic NLP, multi-dimensional heuristic scoring, behavioral signal analysis, and fraud detection.

---

## Overview

The **Redrob Intelligence Engine** processes 100,000+ candidates from a JSONL dataset and ranks the top candidates against a Job Description using:

- **Semantic TF-IDF similarity** — full profile vs. JD cosine similarity
- **8-dimension scoring** — skills, experience, career, behavioral signals, availability, company quality, education, location
- **Trust & safety layer** — 6 disqualification checks + 4 honeypot/fraud detectors
- **Explainable reasoning** — every ranking decision is score-gated, zero LLM hallucination

---

## Project Structure

```
redrob-talent-intelligence/
├── engine.py          # Core intelligence engine (scoring, ranking, fraud detection)
├── run_ranking.py     # Entry point — loads data, runs engine, writes outputs
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/redrob-talent-intelligence.git
cd redrob-talent-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the data file
Download `candidates.jsonl` from the challenge data source and place it at:
```
data/candidates.jsonl
```
Or update the path in `run_ranking.py`:
```python
candidates_file = Path("path/to/your/candidates.jsonl")
```

---

## Usage

```bash
python run_ranking.py
```

This will:
1. Load all candidates from `candidates.jsonl`
2. Parse the embedded Job Description
3. Run TF-IDF semantic similarity across all candidates
4. Score each candidate across 8 dimensions
5. Apply disqualification & fraud detection
6. Output `submission.csv` and `results.js`

### Change number of top candidates

In `run_ranking.py`, modify:
```python
rankings = engine.rank_candidates(top_n=100)   # Change to 1000, etc.
```

---

## Output Files

| File | Description |
|---|---|
| `submission.csv` | Ranked candidates: `candidate_id, rank, score, reasoning` |
| `results.js` | Full breakdown per candidate for UI dashboard |

---

## Scoring Architecture

| Dimension | Weight | Algorithm |
|---|---|---|
| Skill Match | 25% | 40% heuristic keyword + 60% TF-IDF cosine |
| Behavioral Signals | 25% | Additive weighted sum (10 platform signals) |
| Availability | 15% | Rule-based (open-to-work, notice period, last active) |
| Experience Match | 12% | Piecewise linear penalty around 5–9 year range |
| Career Progression | 10% | Narrative rule engine (tenure, seniority, industry) |
| Company Quality | 8% | Categorical (product vs. consulting) + recency weight |
| Education | 3% | Tiered scoring + Hidden Gem uplift |
| Location + Commute | 2% | Geo-zone × work mode compatibility |

---

## Key Features

- **Zero LLM in reasoning layer** — no hallucination, fully deterministic
- **Honeypot detection** — catches inflated/fraudulent profiles
- **Hidden Gem detection** — surfaces Tier-3/4 college grads with strong alternate signals
- **Batch semantic scoring** — TF-IDF runs once across all N candidates simultaneously
- **No GPU required** — runs entirely on CPU via scikit-learn sparse matrices
- **Fully offline** — no API calls, no cloud dependency

---

## Requirements

- Python 3.8+
- scikit-learn
- openpyxl (optional, for Excel export)

---

## Results (on 100K candidate pool)

```
Candidates processed : 100,000
Top 100 ranked       : ~2 min runtime
Score range (Top 100): 73.69 – 88.77
Hidden Gems rescued  : 13 / 100
```

---

## Architecture

```
JD Text + candidates.jsonl
         │
         ▼
    JD Parsing (regex)
         │
         ▼
    TF-IDF Semantic Similarity (batch)
         │
         ▼
    8-Dimension Scoring (per candidate)
         │
         ▼
    Disqualification + Honeypot Detection
         │
         ▼
    Weighted Score Fusion
         │
         ▼
    submission.csv + results.js
```

---

## License

MIT License — feel free to use, modify and distribute.

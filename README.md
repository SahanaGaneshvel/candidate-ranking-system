# Candidate Ranking System

A hybrid semantic + structured scoring system for ranking 100k candidate profiles against a job description. Built for the Redrob AI Hackathon.

The system combines embedding-based semantic similarity with hand-crafted feature scoring derived from explicit JD criteria, applies behavioral modifiers for availability signals, and filters impossible profiles via honeypot detection.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE GPU STEP (Colab)                             │
│                                                                             │
│  src/precompute.py                                                          │
│    ├─ Load BGE model (BAAI/bge-small-en-v1.5)                               │
│    ├─ Stream 100k candidates from candidates.jsonl                         │
│    ├─ Build text documents (headline + summary + career + education)       │
│    │   └─ Excludes skills list (anti-stuffing measure)                     │
│    ├─ Embed all candidates in batches (L2-normalized)                      │
│    └─ Save: cache/candidate_embeddings.npy + cache/candidate_ids.npy       │
│            models/bge-small-en-v1.5/ (for offline use)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               ONLINE CPU STEP (timed, <5 min, no network)                   │
│                                                                             │
│  rank.py                                                                    │
│    ├─ Load JD from .docx, embed with BGE query prefix                       │
│    ├─ Load precomputed embeddings from cache/                               │
│    ├─ Compute semantic similarity (dot product of L2-normed vectors)        │
│    ├─ Load candidate records for feature scoring                            │
│    ├─ honeypot.py: Filter arithmetically impossible profiles                │
│    ├─ src/features.py: Compute 7 structured feature scores                  │
│    │   ├─ Real ML Experience (0.30) - career history keyword matching       │
│    │   ├─ Title Relevance (0.15) - current/recent job titles                │
│    │   ├─ Product vs Services (0.15) - company type classification          │
│    │   ├─ Domain Match (0.15) - NLP/IR vs CV/speech/robotics                │
│    │   ├─ Experience Fit (0.10) - years of experience (peak 6-8)            │
│    │   ├─ Skills Match (0.10) - validated skills with anti-stuffing         │
│    │   └─ Location Fit (0.05) - India cities preferred                      │
│    ├─ Apply penalties (job hopping, pure academic, recent AI only)          │
│    ├─ Apply behavioral modifiers (activity, response rate, notice period)   │
│    ├─ Combine: final = (0.35*semantic + 0.65*structured) * behavioral       │
│    ├─ Rank by score, output top 100                                         │
│    └─ Write submission.csv with per-candidate reasoning                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place dataset files

```
data/
├── candidates.jsonl      # 100k candidate records (~487MB)
├── job_description.docx  # The job description to match against
└── sample_submission.csv # Example output format
```

### 3. Precompute embeddings (GPU recommended)

Run in Google Colab or any GPU environment:

```bash
python src/precompute.py --candidates data/candidates.jsonl
```

This creates:
- `cache/candidate_embeddings.npy` (~147 MB)
- `cache/candidate_ids.npy`
- `models/bge-small-en-v1.5/` (for offline use)

### 4. Run ranking (CPU, offline, <5 min)

```bash
python rank.py
```

Outputs `submission.csv` with columns: `candidate_id, rank, score, reasoning`

### 5. Validate submission

```bash
python validate_submission.py submission.csv
```

## Evaluation

To evaluate against a labeled dev set:

```bash
python eval/evaluate.py
```

Reports NDCG@10, NDCG@50, MAP, and P@10.

## Design Decisions

**Why exclude skills from embeddings?**
Skills are trivially keyword-stuffed by candidates gaming ATS systems. We handle skills as a structured feature with proficiency/duration validation instead.

**Why 35%/65% semantic/structured split?**
The JD emphasizes structured fit criteria (experience level, company type, domain expertise) over pure semantic similarity. "The right answer involves reasoning about the gap between what the JD says and what the JD means."

**Why use career history for experience instead of profile field?**
The `profile.years_of_experience` field can be inconsistent with documented career durations. Career history `duration_months` sums are more reliable.

**Why honeypot = arithmetic only?**
Behavioral signals (inactivity, low response rate) are treated as ranking modifiers, not disqualifiers. Honeypot detection only flags profiles with impossible data (negative durations, timeline mismatches claiming 20+ years with <5 documented).

**Why BGE-small?**
BAAI/bge-small-en-v1.5 (384-dim) provides strong retrieval performance while keeping embeddings compact (~147 MB for 100k candidates). L2-normalized embeddings enable fast similarity via dot product.

## File Structure

```
├── rank.py                 # Main ranking orchestrator (timed step)
├── honeypot.py             # Arithmetic impossibility detection
├── validate_submission.py  # Official submission validator
├── requirements.txt        # Pinned dependencies
├── src/
│   ├── precompute.py       # Offline embedding generation
│   ├── document.py         # Text document builder
│   ├── features.py         # 7 structured feature scorers
│   └── score.py            # Combined scoring + reasoning
└── eval/
    ├── evaluate.py         # NDCG/MAP evaluation harness
    ├── show.py             # Profile viewer for labeling
    └── devset.csv          # Hand-labeled evaluation set
```

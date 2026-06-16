# Candidate Ranking System

Semantic + honeypot baseline for the Redrob AI Hackathon.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place dataset files in `data/`:
   ```
   data/
   ├── candidates.jsonl      # 100k candidate records (~487MB)
   ├── job_description.docx  # The job description to match against
   └── sample_submission.csv # Example output format
   ```

3. Run precompute (GPU recommended, run in Colab):
   ```bash
   python src/precompute.py
   ```
   This creates:
   - `cache/candidate_embeddings.npy` - Precomputed embeddings
   - `cache/candidate_ids.npy` - Candidate ID mapping
   - `models/bge-small-en-v1.5/` - Saved model for offline use

4. Run ranking (CPU, offline, <5 min):
   ```bash
   python rank.py
   ```
   Outputs `submission.csv`.

5. Validate:
   ```bash
   python validate_submission.py submission.csv
   ```

## Architecture

- **src/document.py** - Builds text documents from candidate profiles for embedding (excludes skills to avoid keyword stuffing)
- **src/precompute.py** - Offline GPU step to embed all 100k candidates
- **rank.py** - CPU-only ranking: embeds JD, computes similarity, applies honeypot filter, outputs top 100
- **honeypot.py** - Detects suspicious/fake candidate profiles

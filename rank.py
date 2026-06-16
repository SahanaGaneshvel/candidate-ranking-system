#!/usr/bin/env python3
"""
Candidate ranking - the timed CPU/no-network step.

This script:
  1. Loads the BGE model from local ./models/ (no network)
  2. Reads job description from .docx, embeds with query prefix
  3. Loads precomputed candidate embeddings
  4. Computes semantic similarity scores
  5. Applies structured feature scoring (title, ML experience, etc.)
  6. Applies honeypot detection to filter suspicious candidates
  7. Combines scores with behavioral modifiers
  8. Outputs top 100 ranked candidates to submission.csv

Hard constraints:
  - Must run in under 5 minutes
  - 16 GB RAM max
  - CPU only, no network access
  - Does NOT embed candidates (uses precomputed embeddings)
"""

import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Set

import numpy as np

# Set offline mode BEFORE importing transformers/sentence-transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer

from honeypot import detect_honeypot
from src.score import rank_candidates_with_scores, generate_reasoning


# BGE query prefix for retrieval tasks
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_job_description(docx_path: str) -> str:
    """
    Load job description text from a .docx file.

    Args:
        docx_path: Path to job_description.docx

    Returns:
        Full text content of the job description
    """
    doc = DocxDocument(docx_path)
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def stream_candidates(jsonl_path: Path, limit_to_ids: Set[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Stream candidates and return a dict mapping candidate_id to full record.

    We need the full record for feature scoring and reasoning generation.
    Streams to avoid memory spike from json.load() on full file.

    Args:
        jsonl_path: Path to candidates.jsonl
        limit_to_ids: If provided, only load candidates with these IDs

    Returns:
        Dict mapping candidate_id to candidate record
    """
    candidates = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            candidate_id = candidate.get("candidate_id", candidate.get("id", ""))
            if candidate_id:
                if limit_to_ids is None or candidate_id in limit_to_ids:
                    candidates[candidate_id] = candidate
    return candidates


def rank_candidates(
    jd_path: str = "data/job_description.docx",
    candidates_path: str = "data/candidates.jsonl",
    embeddings_path: str = "cache/candidate_embeddings.npy",
    ids_path: str = "cache/candidate_ids.npy",
    model_path: str = "models/bge-small-en-v1.5",
    output_path: str = "submission.csv",
    top_k: int = 100,
) -> None:
    """
    Rank candidates against job description and output submission.csv.

    Uses combined semantic + structured scoring with behavioral modifiers.

    Args:
        jd_path: Path to job description .docx
        candidates_path: Path to candidates.jsonl (for feature scoring)
        embeddings_path: Path to precomputed embeddings .npy
        ids_path: Path to candidate IDs .npy
        model_path: Path to saved BGE model
        output_path: Output CSV path
        top_k: Number of top candidates to output
    """
    total_start = time.time()

    # Load model from local path (offline)
    print(f"Loading model from: {model_path}")
    model_start = time.time()
    model = SentenceTransformer(model_path)
    print(f"Model loaded in {time.time() - model_start:.2f}s")

    # Load and embed job description
    print(f"Loading job description from: {jd_path}")
    jd_text = load_job_description(jd_path)
    print(f"Job description length: {len(jd_text)} chars")

    # Embed with BGE query prefix
    print("Embedding job description...")
    jd_with_prefix = BGE_QUERY_PREFIX + jd_text
    jd_embedding = model.encode(
        jd_with_prefix,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize
    )
    jd_vec = jd_embedding.astype(np.float32)
    print(f"JD embedding shape: {jd_vec.shape}")

    # Load precomputed candidate embeddings
    print(f"Loading candidate embeddings from: {embeddings_path}")
    embed_start = time.time()
    candidate_embeddings = np.load(embeddings_path)
    candidate_ids_array = np.load(ids_path, allow_pickle=True)
    candidate_ids = [str(cid) for cid in candidate_ids_array]
    print(f"Loaded {len(candidate_ids):,} candidate embeddings in {time.time() - embed_start:.2f}s")
    print(f"Embeddings shape: {candidate_embeddings.shape}")

    # Compute semantic similarity scores (dot product of normalized vectors = cosine sim)
    print("Computing semantic similarity scores...")
    score_start = time.time()
    semantic_scores = candidate_embeddings @ jd_vec
    print(f"Scores computed in {time.time() - score_start:.4f}s")

    # Load full candidate records for feature scoring
    # Only load candidates we have embeddings for
    embedded_ids = set(candidate_ids)
    print(f"Loading candidate records from: {candidates_path}")
    load_start = time.time()
    candidates = stream_candidates(Path(candidates_path), limit_to_ids=embedded_ids)
    print(f"Loaded {len(candidates):,} candidate records in {time.time() - load_start:.2f}s")

    # Run honeypot detection
    print("Running honeypot detection...")
    honeypot_start = time.time()
    honeypot_ids: Set[str] = set()

    for cid, candidate in candidates.items():
        is_honeypot, reasons, details = detect_honeypot(candidate)
        if is_honeypot:
            honeypot_ids.add(cid)

    print(f"Honeypot detection completed in {time.time() - honeypot_start:.2f}s")
    print(f"Flagged {len(honeypot_ids):,} honeypot candidates")

    # Run combined scoring (semantic + structured + behavioral)
    print("Computing structured feature scores...")
    feature_start = time.time()

    ranked_results = rank_candidates_with_scores(
        candidate_ids=candidate_ids,
        semantic_scores=semantic_scores,
        candidates=candidates,
        honeypot_ids=honeypot_ids,
    )

    print(f"Feature scoring completed in {time.time() - feature_start:.2f}s")

    # Take top K
    top_results = ranked_results[:top_k]

    # Generate output
    print(f"Generating {output_path}...")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (cid, score, breakdown) in enumerate(top_results, start=1):
            candidate = candidates.get(cid, {})
            reasoning = generate_reasoning(candidate, breakdown)
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"RANKING COMPLETE")
    print(f"{'='*60}")
    print(f"Total elapsed time: {total_time:.2f}s")
    print(f"Output written to: {output_path}")
    print(f"Top {top_k} candidates ranked")
    print(f"Honeypots filtered: {len(honeypot_ids):,}")

    # Print top 20 for visual inspection with detailed breakdown
    print(f"\n{'='*60}")
    print("TOP 20 CANDIDATES")
    print(f"{'='*60}")
    print(f"{'Rank':<5}{'ID':<15}{'Score':<8}{'Sem':<6}{'ML':<5}{'Title':<25}{'Concern'}")
    print("-" * 90)

    for rank, (cid, score, breakdown) in enumerate(top_results[:20], start=1):
        candidate = candidates.get(cid, {})
        profile = candidate.get("profile", {}) or {}
        title = profile.get("current_title", "N/A")
        if len(title) > 23:
            title = title[:20] + "..."

        sem_score = breakdown.get("semantic_normalized", 0)
        ml_score = breakdown.get("real_ml_experience", 0)
        penalties = breakdown.get("penalty_reasons", [])
        concern = penalties[0][:20] + "..." if penalties else "-"

        print(f"{rank:<5}{cid:<15}{score:<8.3f}{sem_score:<6.2f}{ml_score:<5.2f}{title:<25}{concern}")

    # Show score distribution
    print(f"\n{'='*60}")
    print("SCORE BREAKDOWN (Top 20)")
    print(f"{'='*60}")
    print(f"{'ID':<15}{'Final':<7}{'Semantic':<9}{'Struct':<8}{'Behav':<6}{'Penalty'}")
    print("-" * 55)

    for rank, (cid, score, breakdown) in enumerate(top_results[:20], start=1):
        sem = breakdown.get("semantic_contribution", 0)
        struct = breakdown.get("structured_contribution", 0)
        behav = breakdown.get("behavioral_modifier", 1.0)
        penalty = breakdown.get("penalties", 0)

        print(f"{cid:<15}{score:<7.3f}{sem:<9.3f}{struct:<8.3f}{behav:<6.2f}{penalty:<.3f}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rank candidates against job description")
    parser.add_argument(
        "--jd", "-j",
        default="data/job_description.docx",
        help="Path to job description .docx"
    )
    parser.add_argument(
        "--candidates", "-c",
        default="data/candidates.jsonl",
        help="Path to candidates.jsonl"
    )
    parser.add_argument(
        "--embeddings", "-e",
        default="cache/candidate_embeddings.npy",
        help="Path to precomputed embeddings"
    )
    parser.add_argument(
        "--ids", "-i",
        default="cache/candidate_ids.npy",
        help="Path to candidate IDs"
    )
    parser.add_argument(
        "--model", "-m",
        default="models/bge-small-en-v1.5",
        help="Path to saved model"
    )
    parser.add_argument(
        "--output", "-o",
        default="submission.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=100,
        help="Number of top candidates to output"
    )

    args = parser.parse_args()

    rank_candidates(
        jd_path=args.jd,
        candidates_path=args.candidates,
        embeddings_path=args.embeddings,
        ids_path=args.ids,
        model_path=args.model,
        output_path=args.output,
        top_k=args.top_k,
    )

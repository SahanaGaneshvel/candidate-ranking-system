#!/usr/bin/env python3
"""
Candidate ranking - the timed CPU/no-network step.

This script:
  1. Loads the BGE model from local ./models/ (no network)
  2. Reads job description from .docx, embeds with query prefix
  3. Loads precomputed candidate embeddings
  4. Computes semantic similarity scores
  5. Applies honeypot detection to filter suspicious candidates
  6. Outputs top 100 ranked candidates to submission.csv

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
from typing import Dict, List, Tuple, Any

import numpy as np

# Set offline mode BEFORE importing transformers/sentence-transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer

from honeypot import detect_honeypot


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


def stream_candidates_for_honeypot(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Stream candidates and return a dict mapping candidate_id to full record.

    We need the full record for honeypot detection and reasoning generation.
    Streams to avoid memory spike from json.load() on full file.

    Args:
        jsonl_path: Path to candidates.jsonl

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
                candidates[candidate_id] = candidate
    return candidates


def generate_reasoning(candidate: Dict[str, Any], score: float) -> str:
    """
    Generate a simple grounded one-liner reasoning from real candidate fields.

    Args:
        candidate: Full candidate record
        score: Semantic similarity score

    Returns:
        Human-readable reasoning string
    """
    profile = candidate.get("profile", {}) or {}

    # Extract key fields
    years_exp = profile.get("years_of_experience")
    current_title = profile.get("current_title", "").strip()
    current_company = profile.get("current_company", "").strip()

    parts = []

    # Add current role info
    if current_title:
        if current_company:
            parts.append(f"{current_title} at {current_company}")
        else:
            parts.append(current_title)

    # Add experience
    if years_exp is not None:
        parts.append(f"{years_exp} years experience")

    # Add score
    parts.append(f"similarity score {score:.3f}")

    if parts:
        return "; ".join(parts)
    else:
        return f"Semantic similarity score: {score:.3f}"


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

    Args:
        jd_path: Path to job description .docx
        candidates_path: Path to candidates.jsonl (for honeypot detection)
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
    candidate_ids = np.load(ids_path, allow_pickle=True)
    print(f"Loaded {len(candidate_ids):,} candidate embeddings in {time.time() - embed_start:.2f}s")
    print(f"Embeddings shape: {candidate_embeddings.shape}")

    # Compute semantic similarity scores (dot product of normalized vectors = cosine sim)
    print("Computing semantic similarity scores...")
    score_start = time.time()
    scores = candidate_embeddings @ jd_vec
    print(f"Scores computed in {time.time() - score_start:.4f}s")

    # Create candidate_id -> score mapping
    id_to_score: Dict[str, float] = {}
    id_to_index: Dict[str, int] = {}
    for i, cid in enumerate(candidate_ids):
        cid_str = str(cid)
        id_to_score[cid_str] = float(scores[i])
        id_to_index[cid_str] = i

    # Load full candidate records for honeypot detection
    print(f"Loading candidates for honeypot detection from: {candidates_path}")
    hp_start = time.time()
    candidates = stream_candidates_for_honeypot(Path(candidates_path))
    print(f"Loaded {len(candidates):,} candidates in {time.time() - hp_start:.2f}s")

    # Run honeypot detection and set flagged candidates' score to -inf
    print("Running honeypot detection...")
    honeypot_start = time.time()
    honeypot_count = 0
    honeypot_reasons: Dict[str, List[str]] = {}

    for cid, candidate in candidates.items():
        is_honeypot, reasons, details = detect_honeypot(candidate)
        if is_honeypot:
            honeypot_count += 1
            honeypot_reasons[cid] = reasons
            if cid in id_to_score:
                id_to_score[cid] = float("-inf")

    print(f"Honeypot detection completed in {time.time() - honeypot_start:.2f}s")
    print(f"Flagged {honeypot_count:,} honeypot candidates")

    # Sort by score descending, then by candidate_id ascending for ties
    print("Sorting candidates...")
    sorted_candidates: List[Tuple[str, float]] = sorted(
        id_to_score.items(),
        key=lambda x: (-x[1], x[0])  # -score for descending, id for ascending tie-break
    )

    # Take top K
    top_candidates = sorted_candidates[:top_k]

    # Generate output
    print(f"Generating {output_path}...")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (cid, score) in enumerate(top_candidates, start=1):
            candidate = candidates.get(cid, {})
            reasoning = generate_reasoning(candidate, score)
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"RANKING COMPLETE")
    print(f"{'='*60}")
    print(f"Total elapsed time: {total_time:.2f}s")
    print(f"Output written to: {output_path}")
    print(f"Top {top_k} candidates ranked")
    print(f"Honeypots filtered: {honeypot_count:,}")

    # Print top 20 for visual inspection
    print(f"\n{'='*60}")
    print("TOP 20 CANDIDATES")
    print(f"{'='*60}")
    print(f"{'Rank':<6}{'Candidate ID':<20}{'Score':<12}{'Title'}")
    print("-" * 70)

    for rank, (cid, score) in enumerate(top_candidates[:20], start=1):
        candidate = candidates.get(cid, {})
        profile = candidate.get("profile", {}) or {}
        title = profile.get("current_title", "N/A")
        if len(title) > 30:
            title = title[:27] + "..."
        print(f"{rank:<6}{cid:<20}{score:<12.4f}{title}")

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

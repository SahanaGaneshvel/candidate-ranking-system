#!/usr/bin/env python3
"""
Evaluation harness for candidate ranking.

Computes NDCG@k, MAP, and P@k against a hand-labeled dev set.
Supports weight sweeping to find optimal parameters.
"""

import argparse
import csv
import json
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_devset(devset_path: str) -> Dict[str, int]:
    """
    Load hand-labeled dev set.

    Args:
        devset_path: Path to CSV with candidate_id,relevance columns

    Returns:
        Dict mapping candidate_id to relevance score (0-4)
    """
    labels = {}
    with open(devset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["candidate_id"].strip()
            rel = int(row["relevance"])
            if rel < 0 or rel > 4:
                raise ValueError(f"Relevance must be 0-4, got {rel} for {cid}")
            labels[cid] = rel
    return labels


def dcg_at_k(relevances: List[int], k: int) -> float:
    """
    Compute DCG@k with standard gain: 2^rel - 1.

    Args:
        relevances: List of relevance scores in ranked order
        k: Cutoff position

    Returns:
        DCG score
    """
    relevances = relevances[:k]
    if not relevances:
        return 0.0

    gains = np.array([2**rel - 1 for rel in relevances])
    # Positions are 1-indexed, so discounts are log2(2), log2(3), ...
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """
    Compute NDCG@k.

    Args:
        relevances: List of relevance scores in ranked order
        k: Cutoff position

    Returns:
        NDCG score (0-1)
    """
    dcg = dcg_at_k(relevances, k)
    # Ideal DCG: sort by relevance descending
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(relevances: List[int], threshold: int = 1) -> float:
    """
    Compute Average Precision.

    Args:
        relevances: List of relevance scores in ranked order
        threshold: Minimum relevance to count as relevant (default 1)

    Returns:
        AP score
    """
    relevant_count = 0
    precision_sum = 0.0

    for i, rel in enumerate(relevances):
        if rel >= threshold:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i

    total_relevant = sum(1 for r in relevances if r >= threshold)
    if total_relevant == 0:
        return 0.0

    return precision_sum / total_relevant


def precision_at_k(relevances: List[int], k: int, threshold: int = 1) -> float:
    """
    Compute Precision@k.

    Args:
        relevances: List of relevance scores in ranked order
        k: Cutoff position
        threshold: Minimum relevance to count as relevant

    Returns:
        P@k score
    """
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    relevant = sum(1 for r in relevances if r >= threshold)
    return relevant / len(relevances)


def compute_metrics(
    ranked_ids: List[str],
    labels: Dict[str, int],
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        ranked_ids: List of candidate IDs in ranked order (best first)
        labels: Dict mapping candidate_id to relevance (0-4)

    Returns:
        Dict with NDCG@10, NDCG@50, MAP, P@10
    """
    # Filter to only labeled candidates, preserving rank order
    labeled_ranked = [cid for cid in ranked_ids if cid in labels]
    relevances = [labels[cid] for cid in labeled_ranked]

    return {
        "NDCG@10": ndcg_at_k(relevances, 10),
        "NDCG@50": ndcg_at_k(relevances, 50),
        "MAP": average_precision(relevances),
        "P@10": precision_at_k(relevances, 10),
    }


def run_ranker(
    candidates: Dict[str, Dict[str, Any]],
    candidate_ids: List[str],
    semantic_scores: np.ndarray,
    feature_cache: Dict[str, Dict[str, Any]],
    w_sem: float,
    w_struct: float,
    feature_weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float]]:
    """
    Run the ranker with given weights.

    Args:
        candidates: Dict of candidate_id -> candidate record
        candidate_ids: List of candidate IDs (same order as semantic_scores)
        semantic_scores: Raw semantic similarity scores
        feature_cache: Pre-computed feature scores per candidate
        w_sem: Semantic weight
        w_struct: Structured weight
        feature_weights: Optional custom feature weights

    Returns:
        List of (candidate_id, score) sorted by score descending
    """
    from src.score import normalize_semantic_scores

    # Default feature weights
    if feature_weights is None:
        feature_weights = {
            "title_relevance": 0.15,
            "real_ml_experience": 0.30,
            "experience_fit": 0.10,
            "product_vs_services": 0.15,
            "domain_match": 0.15,
            "location_fit": 0.05,
            "skills_match": 0.10,
        }

    # Normalize semantic scores
    semantic_normalized = normalize_semantic_scores(semantic_scores)

    results = []

    for i, cid in enumerate(candidate_ids):
        if cid not in feature_cache:
            continue

        fs = feature_cache[cid]

        # Compute structured total with custom weights
        structured_total = sum(
            feature_weights.get(f, 0) * fs.get(f, 0)
            for f in feature_weights
        )

        # Combined score
        base_score = w_sem * semantic_normalized[i] + w_struct * structured_total
        modified_score = base_score * fs.get("behavioral_modifier", 1.0)
        final_score = modified_score - fs.get("penalties", 0)

        results.append((cid, final_score))

    # Sort by score descending, candidate_id ascending for ties
    results.sort(key=lambda x: (-x[1], x[0]))
    return results


def sweep_weights(
    candidates: Dict[str, Dict[str, Any]],
    candidate_ids: List[str],
    semantic_scores: np.ndarray,
    feature_cache: Dict[str, Dict[str, Any]],
    labels: Dict[str, int],
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Grid search over weight combinations.

    Args:
        candidates: Candidate records
        candidate_ids: Candidate IDs
        semantic_scores: Semantic similarity scores
        feature_cache: Cached feature scores
        labels: Dev set labels
        verbose: Print progress

    Returns:
        List of dicts with weights and metrics, sorted by NDCG@10
    """
    # Weight grids to try
    sem_struct_ratios = [
        (0.2, 0.8),
        (0.3, 0.7),
        (0.35, 0.65),
        (0.4, 0.6),
        (0.5, 0.5),
    ]

    # Feature weight variations
    # Key insight: JD emphasizes hands-on ML experience over seniority
    ml_exp_weights = [0.25, 0.30, 0.35, 0.40]
    title_weights = [0.10, 0.15, 0.20]
    product_weights = [0.10, 0.15, 0.20]

    results = []
    total_combos = len(sem_struct_ratios) * len(ml_exp_weights) * len(title_weights) * len(product_weights)

    if verbose:
        print(f"Sweeping {total_combos} weight combinations...")

    for i, ((w_sem, w_struct), ml_w, title_w, prod_w) in enumerate(
        product(sem_struct_ratios, ml_exp_weights, title_weights, product_weights)
    ):
        # Remaining weight to distribute
        remaining = 1.0 - ml_w - title_w - prod_w

        # Skip invalid combinations
        if remaining < 0.15:  # Need at least 0.15 for other features
            continue

        # Distribute remaining among other features
        feature_weights = {
            "title_relevance": title_w,
            "real_ml_experience": ml_w,
            "experience_fit": remaining * 0.25,
            "product_vs_services": prod_w,
            "domain_match": remaining * 0.35,
            "location_fit": remaining * 0.15,
            "skills_match": remaining * 0.25,
        }

        ranked = run_ranker(
            candidates, candidate_ids, semantic_scores,
            feature_cache, w_sem, w_struct, feature_weights
        )
        ranked_ids = [cid for cid, _ in ranked]
        metrics = compute_metrics(ranked_ids, labels)

        results.append({
            "w_sem": w_sem,
            "w_struct": w_struct,
            "ml_exp_weight": ml_w,
            "title_weight": title_w,
            "product_weight": prod_w,
            **metrics,
        })

        if verbose and (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{total_combos}...")

    # Sort by NDCG@10 descending
    results.sort(key=lambda x: -x["NDCG@10"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ranking against dev set")
    parser.add_argument(
        "--devset", "-d",
        default="eval/devset.csv",
        help="Path to dev set CSV (candidate_id,relevance)"
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
        "--sweep",
        action="store_true",
        help="Run weight sweep to find optimal parameters"
    )
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=5,
        help="Number of top weight settings to show in sweep mode"
    )

    args = parser.parse_args()

    # Check devset exists
    if not Path(args.devset).exists():
        print(f"ERROR: Dev set not found at {args.devset}")
        print("Create a CSV with columns: candidate_id,relevance")
        print("Relevance scores: 0=honeypot/irrelevant, 4=ideal candidate")
        sys.exit(1)

    print("Loading dev set labels...")
    labels = load_devset(args.devset)
    print(f"Loaded {len(labels)} labeled candidates")

    # Distribution of labels
    label_counts = {}
    for rel in labels.values():
        label_counts[rel] = label_counts.get(rel, 0) + 1
    print("Label distribution:", dict(sorted(label_counts.items())))

    print("\nLoading candidate embeddings...")
    semantic_scores_raw = np.load(args.embeddings)
    candidate_ids = [str(cid) for cid in np.load(args.ids, allow_pickle=True)]
    print(f"Loaded {len(candidate_ids)} embeddings")

    # Load JD and compute semantic scores
    print("Computing semantic scores against JD...")
    from docx import Document as DocxDocument
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("models/bge-small-en-v1.5")
    doc = DocxDocument("data/job_description.docx")
    jd_text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())

    jd_embedding = model.encode(
        "Represent this sentence for searching relevant passages: " + jd_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    semantic_scores = semantic_scores_raw @ jd_embedding

    # Load only labeled candidates for speed
    print("Loading labeled candidate records...")
    labeled_ids = set(labels.keys())
    candidates = {}
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cid = c.get("candidate_id", "")
            if cid in labeled_ids:
                candidates[cid] = c

    print(f"Loaded {len(candidates)} candidate records")

    # Compute feature scores only for labeled candidates (sufficient for eval)
    print("Computing feature scores (caching)...")
    from src.features import score_features

    feature_cache = {}
    for cid in labeled_ids:
        if cid in candidates:
            feature_cache[cid] = score_features(candidates[cid])

    print(f"Cached features for {len(feature_cache)} candidates")

    if args.sweep:
        print("\n" + "=" * 60)
        print("WEIGHT SWEEP MODE")
        print("=" * 60)

        start = time.time()
        results = sweep_weights(
            candidates, candidate_ids, semantic_scores,
            feature_cache, labels
        )
        elapsed = time.time() - start

        print(f"\nSweep completed in {elapsed:.1f}s")
        print(f"\nTop {args.top} weight settings by NDCG@10:")
        print("-" * 80)
        print(f"{'W_SEM':<7}{'W_STR':<7}{'ML_EXP':<8}{'TITLE':<7}{'PROD':<7}{'NDCG@10':<9}{'NDCG@50':<9}{'MAP':<7}{'P@10'}")
        print("-" * 80)

        for r in results[:args.top]:
            print(
                f"{r['w_sem']:<7.2f}{r['w_struct']:<7.2f}{r['ml_exp_weight']:<8.2f}"
                f"{r['title_weight']:<7.2f}{r['product_weight']:<7.2f}"
                f"{r['NDCG@10']:<9.4f}{r['NDCG@50']:<9.4f}{r['MAP']:<7.4f}{r['P@10']:.4f}"
            )

    else:
        # Run with current weights
        print("\n" + "=" * 60)
        print("EVALUATION WITH CURRENT WEIGHTS")
        print("=" * 60)

        from src.score import W_SEMANTIC, W_STRUCTURED
        print(f"W_SEMANTIC: {W_SEMANTIC}, W_STRUCTURED: {W_STRUCTURED}")

        ranked = run_ranker(
            candidates, candidate_ids, semantic_scores,
            feature_cache, W_SEMANTIC, W_STRUCTURED
        )
        ranked_ids = [cid for cid, _ in ranked]

        metrics = compute_metrics(ranked_ids, labels)

        print("\nMetrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        # Show where labeled candidates ranked
        print("\nLabeled candidates in ranking:")
        labeled_in_ranking = [(i, cid, score) for i, (cid, score) in enumerate(ranked) if cid in labels]
        print(f"{'Rank':<8}{'ID':<18}{'Score':<10}{'Relevance':<10}{'Title'}")
        print("-" * 80)
        for rank, cid, score in labeled_in_ranking[:20]:
            rel = labels[cid]
            title = candidates.get(cid, {}).get("profile", {}).get("current_title", "N/A")[:30]
            print(f"{rank+1:<8}{cid:<18}{score:<10.4f}{rel:<10}{title}")


if __name__ == "__main__":
    main()

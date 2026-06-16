"""
Combined scoring system for candidate ranking.

Combines semantic similarity with structured feature scores,
applies behavioral modifiers, and generates explainable reasoning.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

from src.features import score_features


# =============================================================================
# SCORING WEIGHTS - Tunable constants
# =============================================================================

# JD emphasizes structured fit over pure semantic matching
# "The right answer involves reasoning about the gap between what the JD says
#  and what the JD means"
W_SEMANTIC: float = 0.35  # Semantic similarity weight
W_STRUCTURED: float = 0.65  # Structured feature weight

# Feature weights within structured score (defined in features.py)
# Replicated here for reference:
# - real_ml_experience: 0.30 (most important)
# - title_relevance: 0.15
# - product_vs_services: 0.15
# - domain_match: 0.15
# - experience_fit: 0.10
# - skills_match: 0.10
# - location_fit: 0.05


def normalize_semantic_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize semantic similarity scores to 0-1 range across the pool.

    Uses min-max normalization to make scores comparable.
    """
    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        return np.ones_like(scores) * 0.5

    return (scores - min_score) / (max_score - min_score)


def compute_final_score(
    semantic_score_normalized: float,
    feature_scores: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute final combined score for a candidate.

    Formula:
        final = (W_SEM * semantic_norm + W_STRUCT * structured_total)
                * behavioral_modifier - penalties

    Args:
        semantic_score_normalized: Semantic similarity score (0-1)
        feature_scores: Dict from score_features()

    Returns:
        Tuple of (final_score, breakdown_dict)
    """
    structured_total = feature_scores["structured_total"]
    behavioral_mod = feature_scores["behavioral_modifier"]
    penalties = feature_scores["penalties"]

    # Combined base score
    base_score = (
        W_SEMANTIC * semantic_score_normalized +
        W_STRUCTURED * structured_total
    )

    # Apply behavioral modifier
    modified_score = base_score * behavioral_mod

    # Subtract penalties
    final_score = modified_score - penalties

    # Build breakdown for explainability
    breakdown = {
        "semantic_normalized": semantic_score_normalized,
        "semantic_contribution": W_SEMANTIC * semantic_score_normalized,
        "structured_total": structured_total,
        "structured_contribution": W_STRUCTURED * structured_total,
        "base_score": base_score,
        "behavioral_modifier": behavioral_mod,
        "modified_score": modified_score,
        "penalties": penalties,
        "final_score": final_score,
        # Individual feature scores for reasoning
        "title_relevance": feature_scores["title_relevance"],
        "title_explanation": feature_scores["title_explanation"],
        "real_ml_experience": feature_scores["real_ml_experience"],
        "ml_explanation": feature_scores["ml_explanation"],
        "experience_fit": feature_scores["experience_fit"],
        "experience_explanation": feature_scores["experience_explanation"],
        "product_vs_services": feature_scores["product_vs_services"],
        "product_explanation": feature_scores["product_explanation"],
        "domain_match": feature_scores["domain_match"],
        "domain_explanation": feature_scores["domain_explanation"],
        "location_fit": feature_scores["location_fit"],
        "location_explanation": feature_scores["location_explanation"],
        "skills_match": feature_scores["skills_match"],
        "skills_explanation": feature_scores["skills_explanation"],
        "penalty_reasons": feature_scores["penalty_reasons"],
        "behavioral_explanation": feature_scores["behavioral_explanation"],
    }

    return final_score, breakdown


def generate_reasoning(
    candidate: Dict[str, Any],
    breakdown: Dict[str, Any],
) -> str:
    """
    Generate explainable reasoning for a candidate's ranking.

    Cites the 2-3 top contributors plus one honest concern when a penalty fired.
    No generic templates, no invented facts.
    """
    profile = candidate.get("profile", {}) or {}
    yoe = profile.get("years_of_experience", 0)
    current_title = profile.get("current_title", "N/A")

    # Collect positive factors with their scores
    positives: List[Tuple[float, str]] = []

    # Title relevance
    if breakdown["title_relevance"] >= 0.7:
        positives.append((breakdown["title_relevance"], breakdown["title_explanation"]))

    # Real ML experience (most important)
    if breakdown["real_ml_experience"] >= 0.5:
        positives.append((breakdown["real_ml_experience"] + 0.3, breakdown["ml_explanation"]))  # Boost priority

    # Experience fit
    if breakdown["experience_fit"] >= 0.7:
        positives.append((breakdown["experience_fit"], breakdown["experience_explanation"]))

    # Product company experience
    if breakdown["product_vs_services"] >= 0.7:
        positives.append((breakdown["product_vs_services"], breakdown["product_explanation"]))

    # Domain match
    if breakdown["domain_match"] >= 0.7:
        positives.append((breakdown["domain_match"], breakdown["domain_explanation"]))

    # Skills match
    if breakdown["skills_match"] >= 0.6:
        positives.append((breakdown["skills_match"], breakdown["skills_explanation"]))

    # Location fit
    if breakdown["location_fit"] >= 0.85:
        positives.append((breakdown["location_fit"] - 0.3, breakdown["location_explanation"]))  # Lower priority

    # Sort by score (priority) and take top 3
    positives.sort(key=lambda x: -x[0])
    top_positives = [p[1] for p in positives[:3]]

    # Build reasoning parts
    parts = []

    # Start with experience/title context
    if yoe:
        parts.append(f"{yoe:.0f} yrs exp")

    # Add top positive factors
    for pos in top_positives:
        # Clean up the explanation
        clean = pos.replace("relevant title: ", "").replace("strong ML background: ", "ML: ")
        clean = clean.replace("strong skills: ", "skills: ").replace("good skills: ", "skills: ")
        clean = clean.replace("strong product company exp: ", "product co: ")
        clean = clean.replace("strong NLP/IR/retrieval focus", "NLP/IR focus")
        parts.append(clean)

    # Add concern if penalties fired
    penalty_reasons = breakdown.get("penalty_reasons", [])
    if penalty_reasons:
        concern = penalty_reasons[0]
        parts.append(f"concern: {concern}")

    # Add behavioral note if significant
    behavioral_mod = breakdown.get("behavioral_modifier", 1.0)
    if behavioral_mod < 0.92:
        parts.append(f"concern: {breakdown.get('behavioral_explanation', 'low engagement')}")

    # Join with semicolons
    reasoning = "; ".join(parts)

    # Ensure not too long
    if len(reasoning) > 200:
        reasoning = reasoning[:197] + "..."

    return reasoning


def rank_candidates_with_scores(
    candidate_ids: List[str],
    semantic_scores: np.ndarray,
    candidates: Dict[str, Dict[str, Any]],
    honeypot_ids: set,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Rank candidates using combined semantic + structured scoring.

    Args:
        candidate_ids: List of candidate IDs (same order as semantic_scores)
        semantic_scores: Raw semantic similarity scores
        candidates: Dict mapping candidate_id to full candidate record
        honeypot_ids: Set of candidate IDs flagged as honeypots

    Returns:
        List of (candidate_id, final_score, breakdown) sorted by score desc,
        ties broken by candidate_id ascending
    """
    # Normalize semantic scores across the pool
    semantic_normalized = normalize_semantic_scores(semantic_scores)

    results = []

    for i, cid in enumerate(candidate_ids):
        # Skip honeypots
        if cid in honeypot_ids:
            continue

        candidate = candidates.get(cid)
        if not candidate:
            continue

        # Compute feature scores
        feature_scores = score_features(candidate)

        # Compute final combined score
        final_score, breakdown = compute_final_score(
            semantic_normalized[i],
            feature_scores,
        )

        # Add raw semantic for reference
        breakdown["semantic_raw"] = float(semantic_scores[i])

        results.append((cid, final_score, breakdown))

    # Sort by score descending, then by candidate_id ascending for ties
    results.sort(key=lambda x: (-x[1], x[0]))

    return results

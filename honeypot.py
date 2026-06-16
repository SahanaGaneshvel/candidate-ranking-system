"""
Honeypot detection for candidate profiles.

Detects arithmetically impossible or logically inconsistent profiles
that indicate fake/fraudulent data.

IMPORTANT: Behavioral signals (response rate, interview completion, activity)
are NEVER honeypot criteria - they are handled separately as ranking modifiers.
Honeypot detection only flags profiles that are impossible, not just undesirable.
"""

from typing import Any, Tuple, List, Dict


def detect_honeypot(candidate: dict) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Detect if a candidate profile is a honeypot (fake/impossible data).

    Only flags profiles with arithmetically impossible or logically
    inconsistent data. Does NOT consider behavioral signals.

    Args:
        candidate: Full candidate record from candidates.jsonl

    Returns:
        Tuple of:
            - is_honeypot: bool, True if candidate should be excluded
            - reasons: List of human-readable reason strings
            - details: Dict with detection metadata
    """
    reasons: List[str] = []
    details: Dict[str, Any] = {}

    profile = candidate.get("profile", {}) or {}
    career_history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    education = candidate.get("education", []) or []

    # === Arithmetic impossibility checks only ===

    # Check 1: Impossible years of experience (schema max is 50)
    yoe = profile.get("years_of_experience")
    if yoe is not None and yoe > 50:
        reasons.append(f"Impossible years of experience: {yoe}")
        details["impossible_yoe"] = yoe

    # Check 2: Negative years of experience
    if yoe is not None and yoe < 0:
        reasons.append(f"Negative years of experience: {yoe}")
        details["negative_yoe"] = yoe

    # Check 3: Career history exceeds schema limit (max 10 per schema)
    if len(career_history) > 10:
        reasons.append(f"Career history exceeds schema limit: {len(career_history)} > 10")
        details["career_history_count"] = len(career_history)

    # Check 4: Education exceeds schema limit (max 5 per schema)
    if len(education) > 5:
        reasons.append(f"Education exceeds schema limit: {len(education)} > 5")
        details["education_count"] = len(education)

    # Check 5: Career history with impossible duration
    for i, job in enumerate(career_history):
        duration = job.get("duration_months")
        if duration is not None and duration < 0:
            reasons.append(f"Negative job duration in career_history[{i}]: {duration}")
            details["negative_duration"] = duration

    # Check 6: Skills with impossible duration
    for i, skill in enumerate(skills):
        duration = skill.get("duration_months")
        if duration is not None and duration < 0:
            reasons.append(f"Negative skill duration: {skill.get('name', 'unknown')}")
            details["negative_skill_duration"] = duration
            break  # One is enough to flag

    # Check 7: Experience years inconsistent with career history
    # If claimed 20+ years but career history sums to <5 years
    if yoe is not None and yoe > 20:
        total_career_months = sum(
            job.get("duration_months", 0) or 0
            for job in career_history
        )
        total_career_years = total_career_months / 12
        if total_career_years < 5 and len(career_history) > 0:
            reasons.append(
                f"Experience mismatch: claims {yoe} years but career history shows {total_career_years:.1f}"
            )
            details["experience_mismatch"] = {
                "claimed": yoe,
                "documented": total_career_years
            }

    is_honeypot = len(reasons) > 0

    return is_honeypot, reasons, details

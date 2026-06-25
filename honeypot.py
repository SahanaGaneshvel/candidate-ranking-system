"""
Honeypot detection for candidate profiles.

Detects arithmetically impossible or logically inconsistent profiles
that indicate fake/fraudulent data.

IMPORTANT: Behavioral signals (response rate, interview completion, activity)
are NEVER honeypot criteria - they are handled separately as ranking modifiers.
Honeypot detection only flags profiles that are impossible, not just undesirable.

This module uses exactly four validated arithmetic checks:
1. Timeline-span gap: years_of_experience vs career span mismatch
2. Role duration mismatch: duration_months vs (end_date - start_date) inconsistency
3. Expert-with-zero-time: advanced/expert skills with 0 duration_months
4. Documented-experience gap: years_of_experience vs sum of role durations
"""

from datetime import datetime
from typing import Any, Tuple, List, Dict


def _parse_date(date_str: str) -> datetime | None:
    """Parse date string to datetime, returning None if invalid."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _months_between(start: datetime, end: datetime) -> float:
    """Calculate months between two dates."""
    delta = end - start
    return delta.days / 30.44  # Average days per month


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

    yoe = profile.get("years_of_experience")

    # =========================================================================
    # Check 1: Timeline-span gap
    # career_span = (latest end_date - earliest start_date) in years
    # Flag if years_of_experience - career_span > 1.5
    # =========================================================================
    if yoe is not None and len(career_history) > 0:
        start_dates = []
        end_dates = []

        for job in career_history:
            start = _parse_date(job.get("start_date"))
            if start:
                start_dates.append(start)

            end = _parse_date(job.get("end_date"))
            if end:
                end_dates.append(end)
            elif job.get("is_current"):
                end_dates.append(datetime.now())

        if start_dates and end_dates:
            earliest_start = min(start_dates)
            latest_end = max(end_dates)
            career_span = _months_between(earliest_start, latest_end) / 12

            gap = yoe - career_span
            if gap > 1.5:
                reasons.append(
                    f"Timeline-span gap: claims {yoe:.1f} yrs but career spans {career_span:.1f} yrs (gap: {gap:.1f})"
                )
                details["timeline_gap"] = {
                    "claimed_yoe": yoe,
                    "career_span": career_span,
                    "gap": gap
                }

    # =========================================================================
    # Check 2: Role duration mismatch
    # For each role, flag if abs(duration_months - months_between(start, end)) > 3
    # =========================================================================
    duration_mismatches = []
    for i, job in enumerate(career_history):
        duration_months = job.get("duration_months")
        if duration_months is None:
            continue

        start = _parse_date(job.get("start_date"))
        end = _parse_date(job.get("end_date"))

        if not start:
            continue

        if not end and job.get("is_current"):
            end = datetime.now()

        if end:
            computed_months = _months_between(start, end)
            diff = abs(duration_months - computed_months)

            if diff > 3:
                duration_mismatches.append({
                    "role_index": i,
                    "title": job.get("title", "Unknown"),
                    "claimed": duration_months,
                    "computed": computed_months,
                    "diff": diff
                })

    if duration_mismatches:
        reasons.append(
            f"Role duration mismatch: {len(duration_mismatches)} role(s) have duration_months inconsistent with dates"
        )
        details["duration_mismatches"] = duration_mismatches

    # =========================================================================
    # Check 3: Expert-with-zero-time
    # Flag if count of skills with proficiency in {advanced, expert} AND
    # duration_months == 0 is >= 3
    # =========================================================================
    expert_zero_skills = []
    for skill in skills:
        proficiency = (skill.get("proficiency") or "").lower()
        duration = skill.get("duration_months")

        if proficiency in {"advanced", "expert"} and duration == 0:
            expert_zero_skills.append(skill.get("name", "Unknown"))

    if len(expert_zero_skills) >= 3:
        reasons.append(
            f"Expert-with-zero-time: {len(expert_zero_skills)} advanced/expert skills with 0 duration"
        )
        details["expert_zero_skills"] = expert_zero_skills

    # =========================================================================
    # Check 4: Documented-experience gap
    # sum_dur = sum(role duration_months) / 12
    # Flag if years_of_experience - sum_dur > 1.5
    # Note: Only flags INFLATED profiles (yoe > documented), not deflated ones
    # =========================================================================
    if yoe is not None and len(career_history) > 0:
        total_months = sum(
            job.get("duration_months", 0) or 0
            for job in career_history
        )
        sum_dur = total_months / 12

        gap = yoe - sum_dur
        if gap > 1.5:
            reasons.append(
                f"Documented-experience gap: claims {yoe:.1f} yrs but roles sum to {sum_dur:.1f} yrs (gap: {gap:.1f})"
            )
            details["documented_gap"] = {
                "claimed_yoe": yoe,
                "documented_yrs": sum_dur,
                "gap": gap
            }

    is_honeypot = len(reasons) > 0

    return is_honeypot, reasons, details

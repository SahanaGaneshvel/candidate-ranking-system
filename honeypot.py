"""
Honeypot detection for candidate profiles.

Identifies suspicious candidates that may be fake, fraudulent, or
artificially inflated profiles that should be filtered from rankings.

Based on the actual candidate schema and redrob_signals fields.
"""

from typing import Any, Tuple, List, Dict
import re
from datetime import datetime, timedelta


def detect_honeypot(candidate: dict) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Detect if a candidate profile is a honeypot (fake/suspicious).

    A honeypot candidate is one who appears legitimate but has signals
    indicating they are:
    - Fake/bot-generated profiles
    - Keyword-stuffed profiles gaming the system
    - Inactive candidates who won't respond
    - Candidates with impossible/inconsistent data

    Args:
        candidate: Full candidate record from candidates.jsonl

    Returns:
        Tuple of:
            - is_honeypot: bool, True if candidate should be flagged
            - reasons: List of human-readable reason strings
            - details: Dict with detection metadata
    """
    reasons: List[str] = []
    details: Dict[str, Any] = {}

    profile = candidate.get("profile", {}) or {}
    career_history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    education = candidate.get("education", []) or []
    redrob_signals = candidate.get("redrob_signals", {}) or {}

    # === Profile-based checks ===

    # Check 1: Impossible years of experience (schema max is 50)
    yoe = profile.get("years_of_experience")
    if yoe is not None and yoe > 50:
        reasons.append(f"Impossible years of experience: {yoe}")
        details["impossible_yoe"] = yoe

    # Check 2: Skills keyword stuffing - excessive skill count
    # Schema has no maxItems for skills, so many skills may indicate stuffing
    if len(skills) > 50:
        reasons.append(f"Excessive skills count: {len(skills)} (likely keyword stuffing)")
        details["skill_count"] = len(skills)

    # Check 3: All skills at expert proficiency with minimal endorsements
    # This suggests self-reported inflation without external validation
    if skills:
        expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
        low_endorsement_experts = [
            s for s in expert_skills
            if s.get("endorsements", 0) == 0
        ]
        if len(expert_skills) == len(skills) and len(skills) > 15:
            if len(low_endorsement_experts) == len(expert_skills):
                reasons.append(
                    f"All {len(skills)} skills marked as expert with zero endorsements"
                )
                details["unvalidated_expert_skills"] = True

    # Check 4: Career history exceeds schema limit (max 10)
    if len(career_history) > 10:
        reasons.append(f"Career history exceeds schema limit: {len(career_history)} > 10")
        details["career_history_count"] = len(career_history)

    # Check 5: Empty or minimal profile content with high experience claims
    summary = profile.get("summary", "") or ""
    headline = profile.get("headline", "") or ""

    has_minimal_content = (
        len(summary) < 50 and
        len(headline) < 20 and
        len(career_history) == 0
    )
    has_high_claims = yoe is not None and yoe > 10

    if has_minimal_content and has_high_claims:
        reasons.append("Minimal profile content with 10+ years experience claim")
        details["minimal_with_high_claims"] = True

    # === Redrob behavioral signal checks ===

    # Check 6: Very low profile completeness with high skill claims
    completeness = redrob_signals.get("profile_completeness_score", 100)
    if completeness < 30 and len(skills) > 20:
        reasons.append(
            f"Low profile completeness ({completeness}%) with {len(skills)} skills"
        )
        details["low_completeness_high_skills"] = True

    # Check 7: No verification signals (email + phone both unverified)
    verified_email = redrob_signals.get("verified_email", True)
    verified_phone = redrob_signals.get("verified_phone", True)

    if verified_email is False and verified_phone is False:
        # Both explicitly unverified - suspicious
        reasons.append("Neither email nor phone verified")
        details["unverified_contact"] = True

    # Check 8: Extremely low interview completion rate (ghosting pattern)
    interview_rate = redrob_signals.get("interview_completion_rate")
    if interview_rate is not None and interview_rate < 0.1:
        reasons.append(f"Very low interview completion rate: {interview_rate:.0%}")
        details["low_interview_completion"] = interview_rate

    # Check 9: Long inactive period (last_active_date very old)
    last_active = redrob_signals.get("last_active_date")
    if last_active:
        try:
            last_active_dt = datetime.strptime(last_active, "%Y-%m-%d")
            # If last active more than 1 year ago
            if (datetime.now() - last_active_dt) > timedelta(days=365):
                reasons.append(f"Inactive for over 1 year (last active: {last_active})")
                details["long_inactive"] = last_active
        except ValueError:
            pass  # Invalid date format, skip this check

    # Check 10: Zero recruiter response rate with high profile views
    # Indicates either fake profile or someone who never responds
    response_rate = redrob_signals.get("recruiter_response_rate", 1.0)
    profile_views = redrob_signals.get("profile_views_received_30d", 0)

    if response_rate == 0 and profile_views > 50:
        reasons.append(
            f"Zero recruiter response rate despite {profile_views} profile views"
        )
        details["zero_response_high_views"] = True

    is_honeypot = len(reasons) > 0

    return is_honeypot, reasons, details

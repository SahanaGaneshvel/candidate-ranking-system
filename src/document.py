"""
Document builder for candidate profiles.

Converts structured candidate data into a single text document suitable
for semantic embedding and similarity search.
"""

from typing import Any, Dict, List, Optional


def build_document(candidate: Dict[str, Any]) -> str:
    """
    Build a text document from a candidate record for semantic embedding.

    Concatenates: headline + summary + career history (title, company, description)
    + education field of study.

    NOTE: We deliberately EXCLUDE the skills list from the embedded text.
    Skills are trivially keyword-stuffed by candidates trying to game ATS systems.
    Including them would cause semantic similarity to be fooled by candidates
    who list every possible skill keyword. Skills should be handled as structured
    features with validation (e.g., checking skill duration, proficiency levels,
    or cross-referencing with career history) rather than as free text for embedding.

    Args:
        candidate: Full candidate record from candidates.jsonl

    Returns:
        Concatenated text document for embedding
    """
    parts: List[str] = []

    profile = candidate.get("profile", {}) or {}

    # Add headline
    headline = profile.get("headline")
    if headline and isinstance(headline, str) and headline.strip():
        parts.append(headline.strip())

    # Add summary
    summary = profile.get("summary")
    if summary and isinstance(summary, str) and summary.strip():
        parts.append(summary.strip())

    # Add career history: "{title} at {company}: {description}"
    career_history = candidate.get("career_history", []) or []
    for role in career_history:
        if not isinstance(role, dict):
            continue

        title = role.get("title", "") or ""
        company = role.get("company", "") or ""
        description = role.get("description", "") or ""

        role_parts: List[str] = []

        if title.strip():
            role_parts.append(title.strip())

        if company.strip():
            if role_parts:
                role_parts.append(f"at {company.strip()}")
            else:
                role_parts.append(company.strip())

        if description.strip():
            if role_parts:
                role_parts.append(f": {description.strip()}")
            else:
                role_parts.append(description.strip())

        if role_parts:
            parts.append(" ".join(role_parts))

    # Add education field of study
    education = candidate.get("education", []) or []
    for edu in education:
        if not isinstance(edu, dict):
            continue

        field_of_study = edu.get("field_of_study")
        if field_of_study and isinstance(field_of_study, str) and field_of_study.strip():
            parts.append(field_of_study.strip())

    # Join all parts with newlines for clear separation
    return "\n".join(parts)

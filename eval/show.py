#!/usr/bin/env python3
"""
Helper script to display candidate profiles for labeling.

Reads devset_template.csv and shows each candidate's full profile
one at a time for efficient labeling against JD criteria.

Usage:
    python eval/show.py                    # Show all candidates
    python eval/show.py --start 40         # Start from candidate 40
    python eval/show.py --id CAND_0001234  # Show specific candidate
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def load_candidates(jsonl_path: str) -> dict:
    """Load all candidates into a dict by ID."""
    candidates = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            candidates[c["candidate_id"]] = c
    return candidates


def format_candidate(candidate: dict, index: int = None, total: int = None) -> str:
    """Format a candidate profile for display."""
    lines = []

    # Header
    cid = candidate["candidate_id"]
    if index is not None and total is not None:
        lines.append(f"\n{'='*80}")
        lines.append(f"CANDIDATE {index}/{total}: {cid}")
        lines.append(f"{'='*80}")
    else:
        lines.append(f"\n{'='*80}")
        lines.append(f"CANDIDATE: {cid}")
        lines.append(f"{'='*80}")

    # Profile basics
    profile = candidate.get("profile", {})
    lines.append(f"\n--- PROFILE ---")
    lines.append(f"Title:      {profile.get('current_title', 'N/A')}")
    lines.append(f"Company:    {profile.get('current_company', 'N/A')} ({profile.get('current_company_size', 'N/A')})")
    lines.append(f"Industry:   {profile.get('current_industry', 'N/A')}")
    lines.append(f"Experience: {profile.get('years_of_experience', 'N/A')} years")
    lines.append(f"Location:   {profile.get('location', 'N/A')}, {profile.get('country', 'N/A')}")

    # Headline
    headline = profile.get("headline", "")
    if headline:
        lines.append(f"\nHeadline: {headline}")

    # Summary
    summary = profile.get("summary", "")
    if summary:
        lines.append(f"\nSummary:")
        # Word wrap at ~78 chars
        words = summary.split()
        current_line = "  "
        for word in words:
            if len(current_line) + len(word) + 1 > 78:
                lines.append(current_line)
                current_line = "  " + word
            else:
                current_line += " " + word if current_line != "  " else word
        if current_line.strip():
            lines.append(current_line)

    # Career history - THE MOST IMPORTANT PART FOR LABELING
    career = candidate.get("career_history", [])
    if career:
        lines.append(f"\n--- CAREER HISTORY ({len(career)} roles) ---")
        for i, job in enumerate(career, 1):
            title = job.get("title", "N/A")
            company = job.get("company", "N/A")
            duration = job.get("duration_months", 0)
            is_current = job.get("is_current", False)
            industry = job.get("industry", "")
            company_size = job.get("company_size", "")

            current_marker = " [CURRENT]" if is_current else ""
            lines.append(f"\n  [{i}] {title} @ {company}{current_marker}")
            lines.append(f"      Duration: {duration} months | Industry: {industry} | Size: {company_size}")

            desc = job.get("description", "")
            if desc:
                lines.append(f"      Description:")
                # Word wrap description
                words = desc.split()
                current_line = "        "
                for word in words:
                    if len(current_line) + len(word) + 1 > 78:
                        lines.append(current_line)
                        current_line = "        " + word
                    else:
                        current_line += " " + word if current_line != "        " else word
                if current_line.strip():
                    lines.append(current_line)

    # Education
    education = candidate.get("education", [])
    if education:
        lines.append(f"\n--- EDUCATION ---")
        for edu in education:
            degree = edu.get("degree", "")
            field = edu.get("field_of_study", "")
            institution = edu.get("institution", "")
            tier = edu.get("tier", "")
            lines.append(f"  {degree} in {field} @ {institution} (Tier: {tier})")

    # Key skills (just names, not the full list to avoid noise)
    skills = candidate.get("skills", [])
    if skills:
        # Filter to relevant skills
        relevant_keywords = {
            "python", "pytorch", "tensorflow", "ml", "machine learning",
            "nlp", "search", "ranking", "recommendation", "embedding",
            "vector", "faiss", "elasticsearch", "spark", "airflow",
        }
        relevant_skills = [
            s for s in skills
            if any(kw in s.get("name", "").lower() for kw in relevant_keywords)
        ]
        if relevant_skills:
            lines.append(f"\n--- RELEVANT SKILLS ({len(relevant_skills)} of {len(skills)} total) ---")
            for s in relevant_skills[:10]:
                name = s.get("name", "")
                prof = s.get("proficiency", "")
                dur = s.get("duration_months", 0)
                lines.append(f"  {name}: {prof} ({dur} months)")

    # Behavioral signals (brief)
    signals = candidate.get("redrob_signals", {})
    if signals:
        lines.append(f"\n--- REDROB SIGNALS ---")
        lines.append(f"  Open to work: {signals.get('open_to_work_flag', 'N/A')}")
        lines.append(f"  Response rate: {signals.get('recruiter_response_rate', 'N/A')}")
        lines.append(f"  Last active: {signals.get('last_active_date', 'N/A')}")
        lines.append(f"  Notice period: {signals.get('notice_period_days', 'N/A')} days")
        lines.append(f"  Willing to relocate: {signals.get('willing_to_relocate', 'N/A')}")

    # Labeling prompt
    lines.append(f"\n{'='*80}")
    lines.append("LABEL THIS CANDIDATE (0-4):")
    lines.append("  0 = Honeypot/irrelevant (HR, Content Writer, no ML work)")
    lines.append("  1 = Weak fit (wrong domain, minimal ML, services-only)")
    lines.append("  2 = Moderate fit (some ML, wrong experience level, gaps)")
    lines.append("  3 = Good fit (solid ML/ranking work, right experience, minor gaps)")
    lines.append("  4 = Ideal (6-8yr hands-on ML/ranking/retrieval at product company)")
    lines.append(f"{'='*80}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Show candidates for labeling")
    parser.add_argument(
        "--template", "-t",
        default="eval/devset_template.csv",
        help="Path to devset template CSV"
    )
    parser.add_argument(
        "--candidates", "-c",
        default="data/candidates.jsonl",
        help="Path to candidates.jsonl"
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=1,
        help="Start from candidate number (1-indexed)"
    )
    parser.add_argument(
        "--id",
        type=str,
        help="Show specific candidate by ID"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=1,
        help="Number of candidates to show before pausing"
    )

    args = parser.parse_args()

    # Load candidates
    print("Loading candidates...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates")

    # Load template to get IDs to show
    template_ids = []
    with open(args.template, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            template_ids.append(row["candidate_id"])

    print(f"Template has {len(template_ids)} candidates to label")

    # If specific ID requested
    if args.id:
        if args.id in candidates:
            print(format_candidate(candidates[args.id]))
        else:
            print(f"Candidate {args.id} not found")
        return

    # Show candidates starting from --start
    total = len(template_ids)
    shown = 0

    for i, cid in enumerate(template_ids, 1):
        if i < args.start:
            continue

        if cid not in candidates:
            print(f"WARNING: {cid} not found in candidates.jsonl")
            continue

        print(format_candidate(candidates[cid], i, total))
        shown += 1

        if shown >= args.batch:
            try:
                input("\n[Press Enter for next candidate, Ctrl+C to quit]")
                shown = 0
            except KeyboardInterrupt:
                print("\n\nStopped at candidate", i)
                return


if __name__ == "__main__":
    main()

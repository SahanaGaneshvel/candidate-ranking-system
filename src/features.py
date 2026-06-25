"""
Structured feature scoring for candidate ranking.

All thresholds and keyword lists are derived from the JD's explicit fit/anti-fit
criteria and placed in named constants for tunability and defensibility.
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Set, Tuple

# =============================================================================
# CONSTANTS FROM JD - All thresholds and keyword lists are tunable
# =============================================================================

# --- Title Relevance (from JD: "Senior AI Engineer", ML/AI engineering roles) ---
# JD: Looking for applied ML/AI engineer to build ranking/search systems
POSITIVE_TITLE_KEYWORDS: Set[str] = {
    # Core AI/ML titles
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "research engineer", "applied scientist", "nlp engineer", "deep learning",
    "recommendation", "search engineer", "ranking engineer", "retrieval",
    # Software engineering roles that often do ML
    "software engineer", "backend engineer", "platform engineer",
    "data engineer", "analytics engineer", "full stack",
}

# JD: "AI Specialist" without engineering background is often keyword-stuffed
WEAK_POSITIVE_TITLES: Set[str] = {
    "ai specialist", "data analyst", "business intelligence",
    "technical lead", "tech lead", "architect",
}

# JD explicitly says these are NOT fits - non-engineering/non-ML roles
IRRELEVANT_TITLES: Set[str] = {
    # HR/Admin roles
    "hr manager", "human resources", "recruiter", "talent acquisition",
    # Content/Creative roles
    "content writer", "copywriter", "technical writer",
    "graphic designer", "ui designer", "ux designer",
    # Non-software engineering
    "mechanical engineer", "civil engineer", "electrical engineer",
    "chemical engineer", "structural engineer",
    # Sales/Marketing/Business
    "sales executive", "sales manager", "marketing manager",
    "business development", "account manager", "account executive",
    # Finance/Operations
    "accountant", "finance manager", "operations manager",
    "supply chain", "logistics",
    # Support/Service roles
    "customer support", "customer success", "support engineer",
    # Management without tech depth
    "project manager", "program manager", "scrum master",
    "product manager",  # PM can be relevant but title alone isn't enough
    # QA without ML focus
    "qa engineer", "qa analyst", "quality assurance", "test engineer",
    # Specialized non-ML dev roles
    "java developer", ".net developer", "php developer",
}

# --- Experience Fit (from JD: "5-9 years", peak at 6-8) ---
EXPERIENCE_PEAK_MIN: float = 6.0  # JD: "6-8 years total experience"
EXPERIENCE_PEAK_MAX: float = 8.0
EXPERIENCE_OK_MIN: float = 5.0   # JD: "5-9 years"
EXPERIENCE_OK_MAX: float = 9.0
EXPERIENCE_HARD_MIN: float = 4.0  # JD: below this is penalized
EXPERIENCE_TAPER_START: float = 11.0  # JD: above this tapers

# --- Real ML Experience Keywords (from JD: actual ML/ranking/search work) ---
# JD: "shipped at least one end-to-end ranking, search, or recommendation system"
ML_EXPERIENCE_KEYWORDS: Set[str] = {
    # Core ML/AI work
    "machine learning", "deep learning", "neural network", "model training",
    "model deployment", "ml pipeline", "ml system", "mlops",
    # Retrieval/Ranking/Search - JD's core focus
    "ranking system", "search system", "recommendation system", "retrieval",
    "information retrieval", "vector search", "semantic search",
    "embedding", "sentence transformer", "bert", "transformer",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    # NLP - JD mentions NLP/IR expertise
    "nlp", "natural language", "text classification", "named entity",
    "sentiment analysis", "language model", "llm", "fine-tuning", "fine tuning",
    "rag", "retrieval augmented", "prompt engineering",
    # Data science production work
    "a/b test", "ab test", "experimentation", "feature engineering",
    "model evaluation", "ndcg", "mrr", "precision", "recall",
    "production model", "deployed model", "inference", "serving",
}

# JD: "primarily CV/speech/robotics with no NLP → negative"
CV_SPEECH_ROBOTICS_KEYWORDS: Set[str] = {
    "computer vision", "image classification", "object detection",
    "image segmentation", "opencv", "yolo", "cnn for images",
    "speech recognition", "speech synthesis", "tts", "asr",
    "robotics", "ros", "autonomous", "perception", "lidar",
}

# --- Services/Consulting Companies (from JD: explicit anti-fit) ---
# JD: "People who have only worked at consulting firms (TCS, Infosys, Wipro,
#      Accenture, Cognizant, Capgemini, etc.) in their entire career"
SERVICES_COMPANIES: Set[str] = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
    "mphasis", "ltimindtree", "persistent", "zensar", "cyient",
    "hexaware", "niit", "mastech", "syntel", "virtusa",
    "deloitte", "kpmg", "ey", "ernst young", "pwc",  # Big 4 consulting
}

# Known product companies (positive signal)
PRODUCT_COMPANIES: Set[str] = {
    # Indian product companies
    "flipkart", "swiggy", "zomato", "razorpay", "phonepe", "paytm",
    "ola", "uber", "meesho", "cred", "dream11", "mpl",
    "freshworks", "zoho", "postman", "browserstack", "chargebee",
    "clevertap", "moengage", "haptik", "yellow.ai", "verloop",
    "sharechat", "dailyhunt", "inmobi", "glance", "pratilipi",
    # Global product companies
    "google", "meta", "facebook", "amazon", "microsoft", "apple",
    "netflix", "spotify", "airbnb", "uber", "lyft", "doordash",
    "stripe", "square", "shopify", "atlassian", "slack", "notion",
    "openai", "anthropic", "cohere", "huggingface",
    # Startups mentioned in JD
    "redrob",
}

# --- Location Fit (from JD: Pune/Noida preferred, India cities OK) ---
# JD: "Pune/Noida-preferred", "Hyderabad, Pune, Mumbai, Delhi NCR welcome"
TARGET_LOCATIONS: Set[str] = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "ncr",
    "bengaluru", "bangalore", "gurgaon", "gurugram", "chennai",
}

# JD: "Outside India: case-by-case, but we don't sponsor work visas"
INDIA_COUNTRY: str = "india"

# --- Skills Match (from JD: core technical requirements) ---
# JD: "Production experience with embeddings-based retrieval systems"
# JD: "Production experience with vector databases or hybrid search"
# JD: "Strong Python"
# JD: "Hands-on experience designing evaluation frameworks for ranking systems"
CORE_SKILLS: Set[str] = {
    "python", "pytorch", "tensorflow", "keras",
    "sentence-transformers", "transformers", "huggingface",
    "embeddings", "vector database", "faiss", "pinecone", "weaviate",
    "elasticsearch", "opensearch", "qdrant", "milvus",
    "nlp", "information retrieval", "search", "ranking",
    "recommendation", "machine learning", "deep learning",
    "langchain", "llm", "rag", "fine-tuning",
    "spark", "airflow", "mlflow", "kubeflow",
    "docker", "kubernetes", "aws", "gcp", "azure",
}

# Minimum proficiency/duration to count a skill (anti-stuffing)
SKILL_MIN_PROFICIENCY: Set[str] = {"intermediate", "advanced", "expert"}
SKILL_MIN_DURATION_MONTHS: int = 6

# --- Job Hopping (from JD: "Title-chasers... switching every 1.5 years") ---
JOB_HOP_THRESHOLD_MONTHS: int = 18  # Average tenure below this is penalized

# --- Recent AI-only (from JD: "AI experience only in last 12 months") ---
RECENT_AI_THRESHOLD_MONTHS: int = 12

# --- Behavioral Signals (from JD: availability/responsiveness matters) ---
# JD: "A perfect-on-paper candidate who hasn't logged in for 6 months and has
#      a 5% recruiter response rate is, for hiring purposes, not actually available"
INACTIVE_DAYS_THRESHOLD: int = 180  # 6 months
LOW_RESPONSE_RATE: float = 0.15
HIGH_RESPONSE_RATE: float = 0.50
LONG_NOTICE_DAYS: int = 60  # JD: "We'd love sub-30-day notice"


# =============================================================================
# FEATURE SCORING FUNCTIONS
# =============================================================================

def _normalize_text(text: str) -> str:
    """Lowercase and normalize text for matching."""
    return text.lower().strip()


def _text_contains_any(text: str, keywords: Set[str]) -> bool:
    """Check if text contains any of the keywords."""
    text_lower = _normalize_text(text)
    return any(kw in text_lower for kw in keywords)


def _count_keyword_matches(text: str, keywords: Set[str]) -> int:
    """Count how many keywords appear in text."""
    text_lower = _normalize_text(text)
    return sum(1 for kw in keywords if kw in text_lower)


def score_title_relevance(candidate: Dict[str, Any]) -> Tuple[float, str]:
    """
    Score based on current/recent job title relevance.

    Returns:
        Tuple of (score 0-1, explanation string)
    """
    profile = candidate.get("profile", {}) or {}
    career_history = candidate.get("career_history", []) or []

    current_title = _normalize_text(profile.get("current_title", ""))

    # Check current title first
    if _text_contains_any(current_title, POSITIVE_TITLE_KEYWORDS):
        return 1.0, f"relevant title: {profile.get('current_title', 'N/A')}"

    if _text_contains_any(current_title, WEAK_POSITIVE_TITLES):
        return 0.6, f"somewhat relevant: {profile.get('current_title', 'N/A')}"

    if _text_contains_any(current_title, IRRELEVANT_TITLES):
        # Check if they had relevant titles before
        for job in career_history[:3]:  # Check recent 3 roles
            job_title = _normalize_text(job.get("title", ""))
            if _text_contains_any(job_title, POSITIVE_TITLE_KEYWORDS):
                return 0.4, f"past relevant role, now {profile.get('current_title', 'N/A')}"
        return 0.1, f"unrelated title: {profile.get('current_title', 'N/A')}"

    # Unknown title - check career history
    for job in career_history[:2]:
        job_title = _normalize_text(job.get("title", ""))
        if _text_contains_any(job_title, POSITIVE_TITLE_KEYWORDS):
            return 0.7, f"relevant recent role: {job.get('title', 'N/A')}"

    return 0.3, f"neutral title: {profile.get('current_title', 'N/A')}"


def score_real_ml_experience(candidate: Dict[str, Any]) -> Tuple[float, str]:
    """
    Score based on actual ML work in career history descriptions.

    JD: "The right answer involves reasoning about the gap between what
    the JD says and what the JD means."

    Scans career_history descriptions (NOT profile summary) for evidence
    of actually building ML/ranking/search/recommendation/retrieval systems.
    """
    career_history = candidate.get("career_history", []) or []

    if not career_history:
        return 0.0, "no career history"

    total_ml_matches = 0
    ml_roles = 0
    cv_speech_only = True
    evidence = []

    for job in career_history:
        description = _normalize_text(job.get("description", ""))
        title = job.get("title", "")

        ml_matches = _count_keyword_matches(description, ML_EXPERIENCE_KEYWORDS)
        cv_matches = _count_keyword_matches(description, CV_SPEECH_ROBOTICS_KEYWORDS)

        if ml_matches > 0:
            cv_speech_only = False
            total_ml_matches += ml_matches
            ml_roles += 1
            if ml_matches >= 3:
                evidence.append(f"{title} (strong ML)")
            elif ml_matches >= 1:
                evidence.append(f"{title} (some ML)")

    # Penalize CV/speech/robotics-only profiles
    if cv_speech_only and total_ml_matches == 0:
        profile = candidate.get("profile", {}) or {}
        summary = _normalize_text(profile.get("summary", ""))
        if _count_keyword_matches(summary, CV_SPEECH_ROBOTICS_KEYWORDS) > 2:
            return 0.15, "primarily CV/speech/robotics background"

    if total_ml_matches == 0:
        return 0.1, "no ML evidence in career history"

    # Score based on depth and breadth
    if ml_roles >= 2 and total_ml_matches >= 6:
        return 1.0, f"strong ML background: {', '.join(evidence[:2])}"
    elif ml_roles >= 1 and total_ml_matches >= 3:
        return 0.75, f"solid ML experience: {', '.join(evidence[:2])}"
    elif total_ml_matches >= 2:
        return 0.5, f"some ML work: {', '.join(evidence[:1])}"
    else:
        return 0.25, "minimal ML evidence"


def _get_years_of_experience(candidate: Dict[str, Any]) -> float:
    """
    Get years of experience from career history (more reliable than profile field).

    The profile.years_of_experience field can be inconsistent with actual
    career history. We use career history duration_months as source of truth.
    """
    career_history = candidate.get("career_history", []) or []

    if not career_history:
        # Fall back to profile field if no career history
        profile = candidate.get("profile", {}) or {}
        return float(profile.get("years_of_experience", 0) or 0)

    total_months = sum(
        job.get("duration_months", 0) or 0
        for job in career_history
    )
    return total_months / 12


def score_experience_fit(candidate: Dict[str, Any]) -> Tuple[float, str]:
    """
    Score based on years of experience.

    JD: "5-9 years", with ideal being 6-8 years.

    Uses career history duration as source of truth (profile field can be inconsistent).
    """
    yoe = _get_years_of_experience(candidate)

    if yoe == 0:
        return 0.5, "experience not specified"

    # Peak score for 6-8 years
    if EXPERIENCE_PEAK_MIN <= yoe <= EXPERIENCE_PEAK_MAX:
        return 1.0, f"{yoe:.1f} years (ideal range)"

    # Good score for 5-9 years
    if EXPERIENCE_OK_MIN <= yoe <= EXPERIENCE_OK_MAX:
        return 0.85, f"{yoe:.1f} years (acceptable range)"

    # Below minimum
    if yoe < EXPERIENCE_HARD_MIN:
        score = max(0.1, yoe / EXPERIENCE_HARD_MIN * 0.4)
        return score, f"{yoe:.1f} years (below minimum)"

    # Between hard min and ok min
    if yoe < EXPERIENCE_OK_MIN:
        return 0.6, f"{yoe:.1f} years (slightly junior)"

    # Above ok max but below taper
    if yoe <= EXPERIENCE_TAPER_START:
        return 0.7, f"{yoe:.1f} years (slightly senior)"

    # Above taper - gradual decline
    excess = yoe - EXPERIENCE_TAPER_START
    score = max(0.4, 0.7 - excess * 0.03)
    return score, f"{yoe:.1f} years (very senior)"


def score_product_vs_services(candidate: Dict[str, Any]) -> Tuple[float, str]:
    """
    Score based on product company vs services/consulting experience.

    JD: Explicitly anti-fit for "entire career at consulting firms"
    """
    career_history = candidate.get("career_history", []) or []

    if not career_history:
        return 0.5, "no career history"

    product_roles = 0
    services_roles = 0
    total_roles = len(career_history)

    product_companies_found = []
    services_companies_found = []

    for job in career_history:
        company = _normalize_text(job.get("company", ""))

        is_product = _text_contains_any(company, PRODUCT_COMPANIES)
        is_services = _text_contains_any(company, SERVICES_COMPANIES)

        if is_product:
            product_roles += 1
            product_companies_found.append(job.get("company", ""))
        elif is_services:
            services_roles += 1
            services_companies_found.append(job.get("company", ""))

    # All services - strong penalty (JD explicit)
    if services_roles == total_roles and services_roles > 0:
        return 0.15, f"entire career at services firms"

    # Mostly services
    if services_roles > product_roles and services_roles >= total_roles * 0.7:
        return 0.3, f"mostly services background"

    # Has product experience
    if product_roles > 0:
        if product_roles >= total_roles * 0.5:
            return 1.0, f"strong product company exp: {product_companies_found[0]}"
        else:
            return 0.7, f"some product company exp: {product_companies_found[0]}"

    # Unknown companies - neutral
    return 0.5, "company background unclear"


def score_domain_match(candidate: Dict[str, Any]) -> Tuple[float, str]:
    """
    Score based on domain expertise alignment.

    JD: NLP/IR/retrieval/ranking/recommendation → strong positive
    JD: CV/speech/robotics without NLP → negative
    """
    profile = candidate.get("profile", {}) or {}
    career_history = candidate.get("career_history", []) or []

    # Combine all text for analysis
    all_text = _normalize_text(profile.get("summary", ""))
    all_text += " " + _normalize_text(profile.get("headline", ""))
    for job in career_history:
        all_text += " " + _normalize_text(job.get("description", ""))

    # Count domain keywords
    nlp_ir_matches = _count_keyword_matches(all_text, {
        "nlp", "natural language", "information retrieval", "search",
        "ranking", "recommendation", "retrieval", "embedding", "vector",
        "semantic", "text", "language model", "llm", "rag",
    })

    cv_speech_matches = _count_keyword_matches(all_text, CV_SPEECH_ROBOTICS_KEYWORDS)

    # Strong NLP/IR focus
    if nlp_ir_matches >= 5 and nlp_ir_matches > cv_speech_matches:
        return 1.0, "strong NLP/IR/retrieval focus"

    # Mixed but has NLP
    if nlp_ir_matches >= 2:
        if cv_speech_matches > nlp_ir_matches:
            return 0.5, "mixed domain, some NLP"
        return 0.8, "good NLP/IR background"

    # Primarily CV/speech/robotics
    if cv_speech_matches >= 3 and nlp_ir_matches < 2:
        return 0.2, "primarily CV/speech/robotics domain"

    # No clear domain signal
    return 0.4, "domain expertise unclear"


def score_location_fit(candidate: Dict[str, Any]) -> Tuple[float, str]:
    """
    Score based on location and relocation willingness.

    JD: Pune/Noida preferred, India cities OK, outside India case-by-case
    """
    profile = candidate.get("profile", {}) or {}
    redrob_signals = candidate.get("redrob_signals", {}) or {}

    location = _normalize_text(profile.get("location", ""))
    country = _normalize_text(profile.get("country", ""))
    willing_to_relocate = redrob_signals.get("willing_to_relocate", False)

    # In target Indian cities
    if _text_contains_any(location, TARGET_LOCATIONS):
        return 1.0, f"in target location: {profile.get('location', 'N/A')}"

    # In India but not target city
    if INDIA_COUNTRY in country:
        if willing_to_relocate:
            return 0.85, f"in India, willing to relocate"
        return 0.7, f"in India: {profile.get('location', 'N/A')}"

    # Outside India
    if willing_to_relocate:
        return 0.4, f"outside India but willing to relocate"

    return 0.2, f"outside India: {profile.get('country', 'N/A')}"


def score_skills_match(candidate: Dict[str, Any]) -> Tuple[float, str]:
    """
    Score based on relevant skills with anti-stuffing measures.

    JD: Core skills (Python, embeddings, vector DB, NLP, ranking)
    Only counts skills with sufficient proficiency/duration.
    """
    skills = candidate.get("skills", []) or []

    if not skills:
        return 0.3, "no skills listed"

    matched_skills = []

    for skill in skills:
        name = _normalize_text(skill.get("name", ""))
        proficiency = _normalize_text(skill.get("proficiency", ""))
        duration = skill.get("duration_months", 0) or 0

        # Check if skill matches core skills
        if not any(core in name for core in CORE_SKILLS):
            continue

        # Anti-stuffing: require meaningful proficiency or duration
        if proficiency in SKILL_MIN_PROFICIENCY or duration >= SKILL_MIN_DURATION_MONTHS:
            matched_skills.append(skill.get("name", ""))

    num_matched = len(matched_skills)

    if num_matched >= 8:
        return 1.0, f"strong skills: {', '.join(matched_skills[:3])}"
    elif num_matched >= 5:
        return 0.8, f"good skills: {', '.join(matched_skills[:3])}"
    elif num_matched >= 3:
        return 0.6, f"some relevant skills: {', '.join(matched_skills[:2])}"
    elif num_matched >= 1:
        return 0.4, f"few relevant skills: {matched_skills[0]}"
    else:
        return 0.2, "no validated relevant skills"


def calculate_penalties(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Calculate penalty score for anti-fit signals.

    Returns:
        Tuple of (total penalty 0-1, list of penalty reasons)
    """
    penalties = []
    total_penalty = 0.0

    profile = candidate.get("profile", {}) or {}
    career_history = candidate.get("career_history", []) or []

    # --- Penalty 1: Clearly unrelated current role ---
    current_title = _normalize_text(profile.get("current_title", ""))
    if _text_contains_any(current_title, IRRELEVANT_TITLES):
        # Check if they have ANY ML experience
        has_ml = False
        for job in career_history:
            desc = _normalize_text(job.get("description", ""))
            if _count_keyword_matches(desc, ML_EXPERIENCE_KEYWORDS) >= 2:
                has_ml = True
                break

        if not has_ml:
            total_penalty += 0.25
            penalties.append(f"unrelated role with no ML background")

    # --- Penalty 2: Job hopping ---
    if len(career_history) >= 3:
        total_months = 0
        for job in career_history:
            total_months += job.get("duration_months", 0) or 0

        avg_tenure = total_months / len(career_history) if career_history else 0

        if avg_tenure < JOB_HOP_THRESHOLD_MONTHS and len(career_history) >= 4:
            total_penalty += 0.15
            penalties.append(f"job hopping (avg tenure {avg_tenure:.0f} months)")

    # --- Penalty 3: Pure academic/research with no production ---
    profile_summary = _normalize_text(profile.get("summary", ""))
    has_production_keywords = any(kw in profile_summary for kw in [
        "production", "deployed", "shipped", "built", "launched",
        "real users", "scale", "pipeline", "system",
    ])

    has_academic_keywords = any(kw in profile_summary for kw in [
        "phd", "research", "paper", "publication", "academic",
        "university", "professor", "postdoc",
    ])

    if has_academic_keywords and not has_production_keywords:
        # Check career history for production work
        has_production_work = False
        for job in career_history:
            desc = _normalize_text(job.get("description", ""))
            if any(kw in desc for kw in ["production", "deployed", "shipped", "users"]):
                has_production_work = True
                break

        if not has_production_work:
            total_penalty += 0.2
            penalties.append("pure research, no production experience")

    # --- Penalty 4: Recent AI-only (LangChain-tutorial profile) ---
    recent_ai_only = True
    has_any_ai = False

    for job in career_history:
        desc = _normalize_text(job.get("description", ""))
        duration = job.get("duration_months", 0) or 0
        is_current = job.get("is_current", False)

        ml_matches = _count_keyword_matches(desc, ML_EXPERIENCE_KEYWORDS)

        if ml_matches > 0:
            has_any_ai = True
            # If this role is not current and not recent, they have older AI experience
            if not is_current and duration > RECENT_AI_THRESHOLD_MONTHS:
                recent_ai_only = False
                break
            # If current role but they've been doing it for a while
            if is_current and duration > RECENT_AI_THRESHOLD_MONTHS:
                recent_ai_only = False
                break

    if has_any_ai and recent_ai_only:
        # Check if they mention "learning", "exploring", "courses"
        summary = _normalize_text(profile.get("summary", ""))
        learning_signals = any(kw in summary for kw in [
            "learning", "exploring", "course", "bootcamp", "transition",
            "side project", "self-taught", "recently",
        ])

        if learning_signals:
            total_penalty += 0.2
            penalties.append("AI experience only recent/learning stage")

    return min(total_penalty, 0.5), penalties  # Cap total penalty


def calculate_behavioral_modifier(candidate: Dict[str, Any]) -> Tuple[float, str]:
    """
    Calculate behavioral modifier based on availability signals.

    JD: "A perfect-on-paper candidate who hasn't logged in for 6 months
    and has a 5% recruiter response rate is not actually available"

    Returns multiplier in range ~0.85-1.1 (never the main axis)
    """
    redrob_signals = candidate.get("redrob_signals", {}) or {}

    modifier = 1.0
    factors = []

    # --- Last active date ---
    last_active = redrob_signals.get("last_active_date")
    if last_active:
        try:
            last_active_dt = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = (datetime.now() - last_active_dt).days

            if days_inactive > INACTIVE_DAYS_THRESHOLD:
                modifier -= 0.08
                factors.append("inactive 6+ months")
            elif days_inactive < 30:
                modifier += 0.03
                factors.append("recently active")
        except ValueError:
            pass

    # --- Recruiter response rate ---
    response_rate = redrob_signals.get("recruiter_response_rate")
    if response_rate is not None:
        if response_rate < LOW_RESPONSE_RATE:
            modifier -= 0.07
            factors.append(f"low response rate ({response_rate:.0%})")
        elif response_rate > HIGH_RESPONSE_RATE:
            modifier += 0.03
            factors.append(f"good response rate")

    # --- Open to work flag ---
    open_to_work = redrob_signals.get("open_to_work_flag")
    if open_to_work is True:
        modifier += 0.02
        factors.append("open to work")

    # --- Interview completion rate ---
    interview_rate = redrob_signals.get("interview_completion_rate")
    if interview_rate is not None:
        if interview_rate < 0.3:
            modifier -= 0.05
            factors.append("low interview completion")
        elif interview_rate > 0.8:
            modifier += 0.02

    # --- Notice period ---
    notice_days = redrob_signals.get("notice_period_days")
    if notice_days is not None:
        if notice_days <= 30:
            modifier += 0.02
            factors.append("short notice")
        elif notice_days > LONG_NOTICE_DAYS:
            modifier -= 0.03
            factors.append("long notice period")

    # --- GitHub activity (treat -1 as MISSING/neutral) ---
    github_score = redrob_signals.get("github_activity_score")
    if github_score is not None and github_score >= 0:  # -1 means not linked
        if github_score > 50:
            modifier += 0.02
            factors.append("active GitHub")

    # --- Offer acceptance rate (treat -1 as MISSING/neutral) ---
    offer_rate = redrob_signals.get("offer_acceptance_rate")
    if offer_rate is not None and offer_rate >= 0:  # -1 means no history
        if offer_rate < 0.3:
            modifier -= 0.03

    # Clamp modifier to reasonable range
    modifier = max(0.85, min(1.1, modifier))

    explanation = "; ".join(factors) if factors else "neutral signals"
    return modifier, explanation


def score_features(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute all feature scores for a candidate.

    Returns:
        Dict with:
            - Individual feature scores (0-1)
            - Feature explanations
            - Total penalty
            - Behavioral modifier
            - Structured total (weighted combination)
    """
    # Compute individual features
    title_score, title_exp = score_title_relevance(candidate)
    ml_exp_score, ml_exp_exp = score_real_ml_experience(candidate)
    exp_fit_score, exp_fit_exp = score_experience_fit(candidate)
    product_score, product_exp = score_product_vs_services(candidate)
    domain_score, domain_exp = score_domain_match(candidate)
    location_score, location_exp = score_location_fit(candidate)
    skills_score, skills_exp = score_skills_match(candidate)

    # Compute penalties
    penalty_total, penalty_reasons = calculate_penalties(candidate)

    # Compute behavioral modifier
    behavioral_mod, behavioral_exp = calculate_behavioral_modifier(candidate)

    # Weighted combination of positive features
    # Weights reflect JD priorities: real ML experience > title > domain > others
    weights = {
        "title_relevance": 0.15,
        "real_ml_experience": 0.30,  # Most important
        "experience_fit": 0.10,
        "product_vs_services": 0.15,
        "domain_match": 0.15,
        "location_fit": 0.05,
        "skills_match": 0.10,
    }

    structured_total = (
        weights["title_relevance"] * title_score +
        weights["real_ml_experience"] * ml_exp_score +
        weights["experience_fit"] * exp_fit_score +
        weights["product_vs_services"] * product_score +
        weights["domain_match"] * domain_score +
        weights["location_fit"] * location_score +
        weights["skills_match"] * skills_score
    )

    return {
        # Individual scores
        "title_relevance": title_score,
        "title_explanation": title_exp,
        "real_ml_experience": ml_exp_score,
        "ml_explanation": ml_exp_exp,
        "experience_fit": exp_fit_score,
        "experience_explanation": exp_fit_exp,
        "product_vs_services": product_score,
        "product_explanation": product_exp,
        "domain_match": domain_score,
        "domain_explanation": domain_exp,
        "location_fit": location_score,
        "location_explanation": location_exp,
        "skills_match": skills_score,
        "skills_explanation": skills_exp,
        # Penalties
        "penalties": penalty_total,
        "penalty_reasons": penalty_reasons,
        # Behavioral
        "behavioral_modifier": behavioral_mod,
        "behavioral_explanation": behavioral_exp,
        # Totals
        "structured_total": structured_total,
    }

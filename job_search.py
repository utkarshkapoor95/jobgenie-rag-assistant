"""
job_search.py - Live job search, fit scoring, and skill-gap aggregation.

Pipeline:
1. search_adzuna_jobs()     -> pulls live listings from Adzuna for a role/location
2. score_job_fit()          -> per-job fit score + matched/missing skills (Groq)
3. aggregate_missing_skills() -> counts how often each skill is missing across all jobs
4. generate_learning_links() -> search-query links for each missing skill (never
                                 hardcoded video IDs or course URLs, since those can
                                 go dead or be wrong; a search link always resolves)

Requires ADZUNA_APP_ID and ADZUNA_APP_KEY (free, no credit card:
https://developer.adzuna.com/signup). Free tier is roughly 1,000 calls/month,
25 requests/minute, which is ample for a portfolio demo.
"""

import json
import os
import urllib.parse

import requests
from dotenv import load_dotenv

from rag_pipeline import _call_llm

load_dotenv()

ADZUNA_APP_ID = os.environ.get("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY", "")
ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs"

# Countries Adzuna supports that are relevant to Indian job seekers
# (full Adzuna list has 19; this is the practical subset for this app)
SUPPORTED_COUNTRIES = {
    "in": "India",
    "us": "United States",
    "gb": "United Kingdom",
    "ca": "Canada",
    "au": "Australia",
    "de": "Germany",
    "sg": "Singapore",
}

# India-specific salary brackets (LPA = Lakhs Per Annum).
# Values are the annual INR floor passed to Adzuna's salary_min filter.
INDIA_SALARY_BRACKETS = {
    "₹3-5 LPA": 300_000,
    "₹5-7 LPA": 500_000,
    "₹7-10 LPA": 700_000,
    "₹10-15 LPA": 1_000_000,
    "₹15 LPA+": 1_500_000,
    "No preference": None,
}


def search_adzuna_jobs(
    job_title: str,
    country_code: str,
    city: str = "",
    salary_min: int | None = None,
    results_per_page: int = 10,
) -> list[dict]:
    """
    Search live job listings via Adzuna.

    Returns a list of dicts: title, company, location, salary_min, salary_max,
    redirect_url (the apply link), description.

    Returns an empty list (not an exception) on any API error, so the caller
    can show a clean "no results" message instead of crashing.
    """
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        print("[job_search] ADZUNA_APP_ID / ADZUNA_APP_KEY not set.")
        return []

    if country_code not in SUPPORTED_COUNTRIES:
        print(f"[job_search] Unsupported country code: {country_code}")
        return []

    url = f"{ADZUNA_BASE_URL}/{country_code}/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": results_per_page,
        "what": job_title,
        "content-type": "application/json",
    }
    if city and city.strip().lower() not in ("any", "no preference", ""):
        params["where"] = city.strip()
    if salary_min:
        params["salary_min"] = salary_min

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[job_search] Adzuna request failed: {e}")
        return []

    jobs = []
    for item in data.get("results", []):
        jobs.append(
            {
                "title": item.get("title", "Untitled role"),
                "company": item.get("company", {}).get("display_name", "Unknown company"),
                "location": item.get("location", {}).get("display_name", city or "Not specified"),
                "salary_min": item.get("salary_min"),
                "salary_max": item.get("salary_max"),
                "redirect_url": item.get("redirect_url", ""),
                "description": item.get("description", ""),
            }
        )
    return jobs


def score_job_fit(resume_context: str, job: dict) -> dict:
    """
    Score one job against the resume. Returns a dict:
    {fit_score, matched_skills, missing_skills, one_line_reason}

    Uses a compact JSON-only prompt to keep token usage low, since this runs
    once per job (up to ~10 calls per search on Groq's free tier).
    """
    prompt = f"""You are JobGenie. Compare this candidate's resume against one job listing.

Candidate background:
{resume_context[:1500]}

Job Title: {job.get('title', '')}
Company: {job.get('company', '')}
Job Description:
{job.get('description', '')[:1200]}

Reply ONLY with valid JSON (no markdown, no backticks):
{{
  "fit_score": 75,
  "matched_skills": ["skill1", "skill2", "skill3"],
  "missing_skills": ["skill1", "skill2"],
  "one_line_reason": "Short reason for the score, max 20 words"
}}"""

    raw = _call_llm(prompt).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)
        result.setdefault("fit_score", 50)
        result.setdefault("matched_skills", [])
        result.setdefault("missing_skills", [])
        result.setdefault("one_line_reason", "")
        return result
    except Exception:
        return {
            "fit_score": 50,
            "matched_skills": [],
            "missing_skills": [],
            "one_line_reason": "Could not score this listing.",
        }


def aggregate_missing_skills(scored_jobs: list[dict]) -> list[tuple[str, int]]:
    """
    Count how often each missing skill appears across all scored jobs.
    Returns a list of (skill, count) sorted by count descending.
    """
    counts: dict[str, int] = {}
    for job in scored_jobs:
        for skill in job.get("missing_skills", []):
            key = skill.strip()
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


def generate_learning_links(skill: str) -> dict:
    """
    Generate search-query links for a missing skill. Deliberately NOT hardcoded
    video IDs or specific course URLs, since those can go dead or be wrong.
    A search-query link always resolves to current, relevant results.
    """
    query = urllib.parse.quote(f"{skill} tutorial")
    return {
        "skill": skill,
        "youtube": f"https://www.youtube.com/results?search_query={query}",
        "freecodecamp": f"https://www.freecodecamp.org/news/search?query={urllib.parse.quote(skill)}",
        "coursera": f"https://www.coursera.org/search?query={urllib.parse.quote(skill)}",
    }


def run_job_search_pipeline(
    resume_context: str,
    job_title: str,
    country_code: str,
    city: str,
    salary_min: int | None = None,
    max_jobs_to_score: int = 8,
) -> dict:
    """
    End-to-end pipeline: search -> score each job -> aggregate missing skills
    -> attach learning links. Returns a dict ready for display:
    {jobs: [...with fit_score...], missing_skills_summary: [...], total_found}
    """
    jobs = search_adzuna_jobs(job_title, country_code, city, salary_min)

    if not jobs:
        return {"jobs": [], "missing_skills_summary": [], "total_found": 0}

    jobs_to_score = jobs[:max_jobs_to_score]
    scored_jobs = []
    for job in jobs_to_score:
        score_result = score_job_fit(resume_context, job)
        job_with_score = {**job, **score_result}
        scored_jobs.append(job_with_score)

    scored_jobs.sort(key=lambda j: j.get("fit_score", 0), reverse=True)

    missing_counts = aggregate_missing_skills(scored_jobs)
    top_missing = missing_counts[:5]  # cap at 5 to keep the report readable
    missing_summary = []
    for skill, count in top_missing:
        links = generate_learning_links(skill)
        links["appears_in"] = count
        links["total_jobs"] = len(scored_jobs)
        missing_summary.append(links)

    return {
        "jobs": scored_jobs,
        "missing_skills_summary": missing_summary,
        "total_found": len(jobs),
    }

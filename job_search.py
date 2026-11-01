"""
job_search.py - Live job search, fit scoring, and skill-gap aggregation.

Uses the JSearch API (OpenWeb Ninja), NOT Adzuna. Switched after confirming
Adzuna's free tier is only 25 test calls total before requiring a paid
subscription with a credit card. JSearch's free tier is genuinely free:
200 requests/month, hard limit, no credit card, confirmed directly on
https://www.openwebninja.com/api/jsearch (pricing table, fetched and
verified before writing this).

Pipeline:
1. search_jobs()             -> pulls live listings for a role/location
2. score_job_fit()           -> per-job fit score + matched/missing skills (Groq)
3. aggregate_missing_skills() -> counts how often each skill is missing across jobs
4. generate_learning_links()  -> search-query links for each missing skill (never
                                  hardcoded video IDs or course URLs, since those
                                  can go dead or be wrong; a search link always works)

Requires JSEARCH_API_KEY. Register free, no credit card, at
https://app.openwebninja.com/signup, then subscribe to the Free plan for
the JSearch API specifically at https://app.openwebninja.com/api/jsearch.
"""

import json
import os
import urllib.parse

import requests
from dotenv import load_dotenv

from config import (
    JSEARCH_URL,
    MAX_JOBS_TO_SCORE,
    PROMPT_SCORE_JOB,
    SUPPORTED_COUNTRIES,
)
from rag_pipeline import _call_llm

load_dotenv()

JSEARCH_API_KEY = os.environ.get("JSEARCH_API_KEY", "")


def search_jobs(
    job_title: str,
    country_code: str,
    city: str = "",
) -> list[dict]:
    """
    Search live job listings via JSearch.

    Returns a list of dicts: title, company, location, salary_min, salary_max,
    apply_link, description.

    Returns an empty list (not an exception) on any API error, so the caller
    can show a clean "no results" message instead of crashing.
    """
    if not JSEARCH_API_KEY:
        print("[job_search] JSEARCH_API_KEY not set.")
        return []

    if country_code not in SUPPORTED_COUNTRIES:
        print(f"[job_search] Unsupported country code: {country_code}")
        return []

    query = job_title
    if city and city.strip().lower() not in ("any", "no preference", ""):
        query = f"{job_title} in {city.strip()}"

    params = {
        "query": query,
        "country": country_code,
        "language": "en",
        "num_pages": "1",
    }
    headers = {"x-api-key": JSEARCH_API_KEY}

    try:
        resp = requests.get(JSEARCH_URL, params=params, headers=headers, timeout=15)
        print(f"[job_search] JSearch HTTP status: {resp.status_code}")
        print(f"[job_search] JSearch raw response (first 500 chars): {resp.text[:500]}")
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[job_search] JSearch request failed: {e}")
        return []

    if not isinstance(data, dict):
        print(f"[job_search] Unexpected response type: {type(data).__name__}, value: {str(data)[:300]}")
        return []

    # CONFIRMED response shape from live testing:
    # {"data": {"jobs": [...]}}
    # The jobs list is nested one level deeper than typical docs suggest.
    raw_jobs = data.get("data") or data.get("jobs") or data.get("results") or []
    if not isinstance(raw_jobs, list):
        print(f"[job_search] 'jobs' field was not a list: {type(raw_jobs).__name__}")
        raw_jobs = []

    jobs = []
    for item in raw_jobs:
        if not isinstance(item, dict):
            continue

        location_parts = [
            p for p in [item.get("job_city"), item.get("job_state"), item.get("job_country")] if p
        ]
        location = ", ".join(location_parts) if location_parts else (city or "Not specified")

        jobs.append(
            {
                "title": item.get("job_title", "Untitled role"),
                "company": item.get("employer_name", "Unknown company"),
                "location": location,
                "salary_min": item.get("job_min_salary"),
                "salary_max": item.get("job_max_salary"),
                "apply_link": item.get("job_apply_link", ""),
                "description": item.get("job_description", ""),
            }
        )
    return jobs


def score_job_fit(resume_context: str, job: dict) -> dict:
    """
    Score one job against the resume. Returns a dict:
    {fit_score, matched_skills, missing_skills, one_line_reason}
    """
    prompt = PROMPT_SCORE_JOB.format(
        resume_context=resume_context[:1500],
        job_title=job.get("title", ""),
        company=job.get("company", ""),
        description=job.get("description", "")[:1200],
    )

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
    max_jobs_to_score: int = MAX_JOBS_TO_SCORE,
) -> dict:
    """
    End-to-end pipeline: search -> score each job -> aggregate missing skills
    -> attach learning links.

    Salary filtering removed: most job postings do not include salary data
    (confirmed in OpenWeb Ninja's own FAQ and from live testing). The salary
    question added friction without reliable payoff.
    """
    jobs = search_jobs(job_title, country_code, city)

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
    top_missing = missing_counts[:5]
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

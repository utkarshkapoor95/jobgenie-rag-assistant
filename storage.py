"""
storage.py - Persistent user storage for JobGenie's Job Tracker.

HuggingFace Spaces free-tier local disk is ephemeral: it is wiped on every
restart (confirmed in HF's own docs, huggingface.co/docs/hub/spaces-storage).
Paid persistent storage starts at $5/month, which this project does not need.

Instead this uses HuggingFace's own documented free workaround: a private
HuggingFace Dataset repo as a JSON datastore, read and written via the
huggingface_hub library. This is genuinely free, requires no credit card,
and uses the same HF account and token this project already has for the
Space itself (just needs write access to datasets too).

One JSON file per user, keyed by a manual ID (an email or name the user
types in). A manual ID was chosen over a resume-content hash because a
hash breaks the moment the user edits even one line of their resume, which
makes it unreliable to demo live to a recruiter.

Storage layout inside the dataset repo:
    users/<sanitized_id>.json

Each file:
{
    "user_id": "someone@example.com",
    "first_seen": "2026-11-02T10:00:00",
    "last_seen": "2026-11-02T10:00:00",
    "searches": [
        {
            "job_title": "Data Analyst",
            "country": "in",
            "city": "Gwalior",
            "searched_at": "2026-11-02T10:00:00",
            "jobs": [
                {
                    "job_key": "<company>::<title>",
                    "title": "...",
                    "company": "...",
                    "location": "...",
                    "fit_score": 68,
                    "apply_link": "...",
                    "status": "new"   # new | applied | interested | not_interested
                }
            ]
        }
    ]
}
"""

import json
import os
import re
from datetime import datetime, timezone

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

DATASET_REPO_ID = os.environ.get("JOBGENIE_DATASET_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

_api = None


def _get_api():
    global _api
    if _api is None:
        _api = HfApi(token=HF_TOKEN)
    return _api


def _sanitize_id(user_id: str) -> str:
    """Turn an email/name into a safe filename."""
    cleaned = re.sub(r"[^a-zA-Z0-9_.@-]", "_", user_id.strip().lower())
    return cleaned[:100] if cleaned else "anonymous"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_configured() -> bool:
    """True if the dataset repo and token are both set up."""
    return bool(DATASET_REPO_ID and HF_TOKEN)


def load_user(user_id: str) -> dict | None:
    """
    Load a user's record from the dataset repo. Returns None if the user
    is new, storage isn't configured, or any error occurs (storage is a
    nice-to-have, never something that should block the core app).
    """
    if not is_configured():
        return None

    safe_id = _sanitize_id(user_id)
    try:
        path = hf_hub_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            filename=f"users/{safe_id}.json",
            token=HF_TOKEN,
        )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    except Exception as e:
        print(f"[storage] load_user failed for {safe_id}: {e}")
        return None


def save_user(user_id: str, record: dict) -> bool:
    """
    Save a user's record back to the dataset repo. Returns True on success,
    False on any failure (caller should degrade gracefully, never crash
    the chat over a storage failure).
    """
    if not is_configured():
        return False

    safe_id = _sanitize_id(user_id)
    record["last_seen"] = _now()

    try:
        api = _get_api()
        content = json.dumps(record, indent=2, ensure_ascii=False).encode("utf-8")
        api.upload_file(
            path_or_fileobj=content,
            path_in_repo=f"users/{safe_id}.json",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message=f"Update record for {safe_id}",
        )
        return True
    except Exception as e:
        print(f"[storage] save_user failed for {safe_id}: {e}")
        return False


def get_or_create_user(user_id: str) -> tuple[dict, bool]:
    """
    Returns (record, is_returning_user).
    Creates a fresh record in memory if the user is new; does not write
    to storage until save_user is called explicitly.
    """
    existing = load_user(user_id)
    if existing is not None:
        return existing, True

    fresh = {
        "user_id": user_id,
        "first_seen": _now(),
        "last_seen": _now(),
        "searches": [],
    }
    return fresh, False


def _job_key(job: dict) -> str:
    """Stable identity for a job listing, used to detect duplicates and new listings."""
    return f"{job.get('company', '')}::{job.get('title', '')}"


def record_search(record: dict, job_title: str, country: str, city: str, scored_jobs: list) -> dict:
    """
    Appends a new search to the user's record. Preserves any existing
    status (applied/interested/not_interested) for jobs seen in a
    previous search, so re-running the same search doesn't wipe history.
    Returns the updated record (caller still needs to call save_user).
    """
    previous_statuses = {}
    for past_search in record.get("searches", []):
        for job in past_search.get("jobs", []):
            previous_statuses[job["job_key"]] = job.get("status", "new")

    job_entries = []
    for job in scored_jobs:
        key = _job_key(job)
        job_entries.append(
            {
                "job_key": key,
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "fit_score": job.get("fit_score", 0),
                "apply_link": job.get("apply_link", ""),
                "status": previous_statuses.get(key, "new"),
            }
        )

    record.setdefault("searches", []).append(
        {
            "job_title": job_title,
            "country": country,
            "city": city,
            "searched_at": _now(),
            "jobs": job_entries,
        }
    )
    return record


def get_all_seen_job_keys(record: dict, exclude_latest: bool = True) -> set:
    """
    Returns the set of job_keys from all searches except the most recent one
    (used to compute 'new since last visit' when exclude_latest=True).
    """
    searches = record.get("searches", [])
    if exclude_latest:
        searches = searches[:-1]
    seen = set()
    for search in searches:
        for job in search.get("jobs", []):
            seen.add(job["job_key"])
    return seen


def update_job_status(record: dict, job_key: str, status: str) -> dict:
    """
    Updates the status of one job across all searches in the record
    (a job the user marked "Applied" stays marked even if it shows up
    again in a later search).
    """
    for search in record.get("searches", []):
        for job in search.get("jobs", []):
            if job["job_key"] == job_key:
                job["status"] = status
    return record


def summarize_tracker(record: dict) -> dict:
    """
    Returns counts by status across all jobs the user has ever seen,
    deduplicated by job_key (latest status wins).
    """
    latest_status = {}
    latest_job_info = {}
    for search in record.get("searches", []):
        for job in search.get("jobs", []):
            latest_status[job["job_key"]] = job.get("status", "new")
            latest_job_info[job["job_key"]] = job

    counts = {"new": 0, "applied": 0, "interested": 0, "not_interested": 0}
    for status in latest_status.values():
        counts[status] = counts.get(status, 0) + 1

    return {
        "counts": counts,
        "total_unique_jobs": len(latest_status),
        "total_searches": len(record.get("searches", [])),
    }

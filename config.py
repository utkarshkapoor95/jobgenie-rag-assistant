"""
config.py - Centralized configuration for JobGenie RAG Assistant.

All constants, model names, prompt templates, and country/API config live here
so the rest of the codebase never hardcodes them.
"""

# ── LLM / Embedding ───────────────────────────────────────────────────────
# NOTE: llama-3.3-70b-versatile was retired by Groq on 2026-08-16.
# openai/gpt-oss-120b is Groq's recommended replacement: same free tier
# shape (30 RPM, 1,000 req/day), actually higher daily token cap (200K vs 100K).
GROQ_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension

# ── Pinecone ──────────────────────────────────────────────────────────────
PINECONE_INDEX_NAME = "jobgenie-index"

# ── JSearch API ───────────────────────────────────────────────────────────
# Free tier: 200 requests/month, no credit card.
# Register at https://app.openwebninja.com/signup
# Subscribe to JSearch Free plan at https://app.openwebninja.com/api/jsearch
JSEARCH_URL = "https://api.openwebninja.com/jsearch/search-v2"

# Countries explicitly confirmed in JSearch documentation FAQ.
SUPPORTED_COUNTRIES = {
    "in": "India",
    "us": "United States",
    "gb": "United Kingdom",
    "ca": "Canada",
    "de": "Germany",
    "sg": "Singapore",
}

# ── Job Search Defaults ───────────────────────────────────────────────────
MAX_JOBS_TO_SCORE = 8

# ── Job Tracker (persistent storage) ──────────────────────────────────────
# Requires two Space secrets: JOBGENIE_DATASET_REPO (e.g. "Utkarsh94123/jobgenie-data")
# and HF_TOKEN (a token with write access to datasets). If either is missing,
# storage.py degrades gracefully and the app runs without memory, same as before.
JOB_STATUS_LABELS = {
    "new": "New",
    "applied": "Applied",
    "interested": "Interested",
    "not_interested": "Not Interested",
}

# ── Skill Detection (quick regex-free check on resume text) ───────────────
SKILL_KEYWORDS = [
    "SQL", "Python", "Power BI", "Excel", "LangChain", "Pinecone",
    "Machine Learning", "Tableau", "Pandas", "NumPy", "RAG", "LLM",
    "n8n", "HubSpot", "Docker", "BigQuery", "Looker", "GCP", "AWS",
    "JavaScript", "React", "TensorFlow", "Scikit-learn", "Git",
]

# ── Prompt Templates ──────────────────────────────────────────────────────

PROMPT_SUGGEST_TITLES = """You are JobGenie, a career advisor. Analyze this resume and suggest exactly 3 job titles that are the best fit for this candidate in the current job market.

Resume:
{resume_text}

Rules:
- Return ONLY valid JSON, no markdown, no backticks, no explanation.
- Pick titles that match the candidate's actual skills and experience level.
- Prefer titles that have high job listing volume (avoid ultra-niche titles like "AI Automation Orchestrator" that return zero search results).
- Good examples: "Data Analyst", "Business Analyst", "Marketing Automation Specialist", "AI Engineer", "GTM Engineer", "Operations Analyst".

Reply format:
{{"titles": ["Title 1", "Title 2", "Title 3"]}}"""


PROMPT_SCORE_JOB = """You are JobGenie. Compare this candidate's resume against one job listing.

Candidate background:
{resume_context}

Job Title: {job_title}
Company: {company}
Job Description:
{description}

Reply ONLY with valid JSON (no markdown, no backticks):
{{
  "fit_score": 75,
  "matched_skills": ["skill1", "skill2", "skill3"],
  "missing_skills": ["skill1", "skill2"],
  "one_line_reason": "Short reason for the score, max 20 words"
}}"""


PROMPT_PARSE_JD = """Extract the following fields from the job description below.
Reply ONLY with valid JSON, no explanation, no markdown, no backticks.

Fields to extract:
- JOB_TITLE
- COMPANY
- EXPERIENCE (e.g. "2-4 years")
- TECHNICAL_SKILLS (comma-separated list)
- SOFT_SKILLS (comma-separated list)

Job Description:
{jd_text}

Reply with JSON only:"""


PROMPT_SKILL_GAP = """You are JobGenie, a career advisor specializing in the Indian job market.

Candidate's background (from resume):
{context}

Job Description:
{jd_text}

Provide a structured skill gap analysis with these exact sections:

### MATCHES - Skills the candidate already has
(List 4-6 specific matches with brief explanation of relevance)

### GAPS - Skills the JD requires that the candidate lacks
(List 3-5 gaps honestly)

### REFRAME - How to position Operations experience as a strength
(For each gap, suggest how ops background partially covers it or how to present it)

### ACTION ITEMS - Top 2-3 things to do before applying
(Specific, actionable steps)

Be honest, specific, and practical. Mention actual skills from the resume."""


PROMPT_PITCH = """You are JobGenie. Write a tailored "Tell me about yourself" pitch for this candidate.

Candidate's background:
{context}

Target Job Description:
{jd_text}

Requirements:
- Exactly 150-180 words
- Start with current status (MBA student transitioning from Operations to Data/AI)
- Mention 2-3 specific skills or projects that match THIS JD
- End with why this specific role excites them
- Tone: confident, professional, natural (not robotic)
- Frame the Operations to AI transition as intentional career evolution
- Suitable for the Indian startup/tech job market

Write ONLY the pitch, no instructions, no labels, no preamble."""


PROMPT_QUESTIONS = """You are JobGenie. Based on this JD and candidate background, predict 10 interview questions.

Candidate's background:
{context}

Job Description:
{jd_text}

Format EXACTLY like this for each question:

**Q1 [Technical]: [Question]**
> Framework: [How to answer using the candidate's actual experience, be specific]

Include:
- 3-4 Technical questions (based on skills in the JD)
- 3-4 Behavioral questions (leadership, teamwork, conflict, failure)
- 2-3 Situational/Case questions (problem-solving scenarios)

Be specific to THIS JD and THIS candidate's background."""


PROMPT_CONFIDENCE = """You are JobGenie. Score this candidate's readiness for the job on a scale of 0-100%.

Candidate's background:
{context}

Job Description:
{jd_text}

Reply in EXACTLY this format:

Technical Skills Match: [0-100]%
Experience Relevance: [0-100]%
Education Fit: [0-100]%
Project Portfolio Match: [0-100]%
Overall Confidence: [0-100]%
SUMMARY: [2-sentence honest assessment, what's strong and what needs work]

Be realistic. Don't inflate scores."""


PROMPT_COMPARE_JD = """You are JobGenie. Evaluate this candidate's fit for the job.

Candidate background:
{context}

Job Description:
{jd_text}

Reply ONLY with valid JSON (no markdown, no backticks):
{{
  "JOB_TITLE": "...",
  "COMPANY": "...",
  "match_pct": 75,
  "top_skills": ["skill1", "skill2", "skill3"],
  "top_gaps": ["gap1", "gap2"],
  "difficulty": "Medium",
  "recommendation": "2-sentence recommendation"
}}"""


PROMPT_QA = """You are JobGenie, a career assistant. Answer based on the candidate's resume.
Be specific and practical for the Indian job market (salary in LPA).

Resume:
{resume_text}

Question: {question}"""

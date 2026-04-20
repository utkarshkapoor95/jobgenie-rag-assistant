"""
rag_pipeline.py — JobGenie RAG Pipeline
All functions called by app.py are defined here.
"""

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ── Globals ────────────────────────────────────────────────────────────────
_pc = None
_index = None
_embedder = None
_llm = None
_resume_context = ""   # in-memory fallback when Pinecone not ready

INDEX_NAME = "jobgenie-index"
EMBED_DIM  = 384        # all-MiniLM-L6-v2 dimension
GROQ_MODEL = "llama-3.3-70b-versatile"


# ── Lazy initializers ──────────────────────────────────────────────────────

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedder


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.4,
            api_key=os.environ.get("GROQ_API_KEY", ""),
        )
    return _llm


def _get_index():
    """Connect to (or create) the Pinecone index."""
    global _pc, _index
    if _index is not None:
        return _index

    api_key = os.environ.get("PINECONE_API_KEY", "")
    if not api_key:
        return None  # graceful degradation — use in-memory fallback

    try:
        _pc = Pinecone(api_key=api_key)
        existing = [i.name for i in _pc.list_indexes()]
        if INDEX_NAME not in existing:
            _pc.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        _index = _pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"[Pinecone] Could not connect: {e}")
        _index = None

    return _index


# ── Public API ─────────────────────────────────────────────────────────────

def ingest_resume_text(resume_text: str) -> None:
    """
    Chunk the resume text and upsert embeddings into Pinecone.
    Falls back to storing in memory if Pinecone is unavailable.
    """
    global _resume_context
    _resume_context = resume_text  # always keep in-memory copy

    index = _get_index()
    embedder = _get_embedder()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_text(resume_text)

    if index is None:
        print("[ingest] Pinecone unavailable — using in-memory context only.")
        return

    try:
        vectors = []
        for i, chunk in enumerate(chunks):
            vec = embedder.embed_query(chunk)
            vectors.append({
                "id": f"resume-{i}",
                "values": vec,
                "metadata": {"text": chunk, "source": "resume"},
            })
        for batch_start in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[batch_start:batch_start + 100])
        print(f"[ingest] Upserted {len(vectors)} chunks into Pinecone.")
    except Exception as e:
        print(f"[ingest] Pinecone upsert failed: {e}")


def _retrieve_context(query: str, top_k: int = 5) -> str:
    """
    Embed query, fetch top_k chunks from Pinecone, return as joined string.
    Falls back to first 3000 chars of in-memory resume.
    """
    index = _get_index()
    embedder = _get_embedder()

    if index is None:
        return _resume_context[:3000]

    try:
        vec = embedder.embed_query(query)
        results = index.query(vector=vec, top_k=top_k, include_metadata=True)
        chunks = [m["metadata"]["text"] for m in results.get("matches", [])]
        return "\n\n".join(chunks) if chunks else _resume_context[:3000]
    except Exception as e:
        print(f"[retrieve] Pinecone query failed: {e}")
        return _resume_context[:3000]


def _call_llm(prompt: str) -> str:
    """Call Groq LLM and return text response."""
    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"LLM error: {str(e)}"


# ── Feature Functions ──────────────────────────────────────────────────────

def parse_jd(jd_text: str) -> dict:
    """
    Extract structured fields from a raw JD string.
    Returns a dict with keys: JOB_TITLE, COMPANY, EXPERIENCE, TECHNICAL_SKILLS, SOFT_SKILLS.
    """
    prompt = f"""Extract the following fields from the job description below.
Reply ONLY with valid JSON — no explanation, no markdown, no backticks.

Fields to extract:
- JOB_TITLE
- COMPANY
- EXPERIENCE (e.g. "2-4 years")
- TECHNICAL_SKILLS (comma-separated list)
- SOFT_SKILLS (comma-separated list)

Job Description:
{jd_text[:3000]}

Reply with JSON only:"""

    raw = _call_llm(prompt)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except Exception:
        return {
            "JOB_TITLE": "Not extracted",
            "COMPANY": "Not extracted",
            "EXPERIENCE": "Not extracted",
            "TECHNICAL_SKILLS": "Not extracted",
            "SOFT_SKILLS": "Not extracted",
        }


def analyze_skill_gap(jd_text: str) -> str:
    """RAG-powered skill gap analysis comparing JD requirements vs resume."""
    context = _retrieve_context(jd_text)
    prompt = f"""You are JobGenie, a career advisor specializing in the Indian job market.

Candidate's background (from resume):
{context}

Job Description:
{jd_text[:2000]}

Provide a structured skill gap analysis with these exact sections:

### MATCHES — Skills the candidate already has
(List 4-6 specific matches with brief explanation of relevance)

### GAPS — Skills the JD requires that the candidate lacks
(List 3-5 gaps honestly)

### REFRAME — How to position Operations experience as a strength
(For each gap, suggest how ops background partially covers it or how to present it)

### ACTION ITEMS — Top 2-3 things to do before applying
(Specific, actionable steps)

Be honest, specific, and practical. Mention actual skills from the resume."""

    return _call_llm(prompt)


def generate_pitch(jd_text: str) -> str:
    """Generate a tailored 60-second elevator pitch."""
    context = _retrieve_context(jd_text)
    prompt = f"""You are JobGenie. Write a tailored "Tell me about yourself" pitch for this candidate.

Candidate's background:
{context}

Target Job Description:
{jd_text[:1500]}

Requirements:
- Exactly 150-180 words
- Start with current status (MBA student transitioning from Operations to Data/AI)
- Mention 2-3 specific skills or projects that match THIS JD
- End with why this specific role excites them
- Tone: confident, professional, natural (not robotic)
- Frame the Operations to AI transition as intentional career evolution
- Suitable for the Indian startup/tech job market

Write ONLY the pitch — no instructions, no labels, no preamble."""

    return _call_llm(prompt)


def predict_questions(jd_text: str) -> str:
    """Predict 10 likely interview questions with answer frameworks."""
    context = _retrieve_context(jd_text)
    prompt = f"""You are JobGenie. Based on this JD and candidate background, predict 10 interview questions.

Candidate's background:
{context}

Job Description:
{jd_text[:1500]}

Format EXACTLY like this for each question:

**Q1 [Technical]: [Question]**
> Framework: [How to answer using the candidate's actual experience — be specific]

Include:
- 3-4 Technical questions (based on skills in the JD)
- 3-4 Behavioral questions (leadership, teamwork, conflict, failure)
- 2-3 Situational/Case questions (problem-solving scenarios)

Be specific to THIS JD and THIS candidate's background."""

    return _call_llm(prompt)


def calculate_confidence(jd_text: str) -> str:
    """Calculate readiness score across 5 dimensions."""
    context = _retrieve_context(jd_text)
    prompt = f"""You are JobGenie. Score this candidate's readiness for the job on a scale of 0-100%.

Candidate's background:
{context}

Job Description:
{jd_text[:1500]}

Reply in EXACTLY this format:

Technical Skills Match: [0-100]%
Experience Relevance: [0-100]%
Education Fit: [0-100]%
Project Portfolio Match: [0-100]%
Overall Confidence: [0-100]%
SUMMARY: [2-sentence honest assessment — what's strong and what needs work]

Be realistic. Don't inflate scores."""

    return _call_llm(prompt)


def compare_jds(jd_list: list) -> list:
    """
    Compare multiple JDs against the candidate's profile.
    Returns a list of dicts with comparison fields.
    """
    results = []
    for jd_text in jd_list:
        context = _retrieve_context(jd_text)
        parsed = parse_jd(jd_text)
        prompt = f"""You are JobGenie. Evaluate this candidate's fit for the job.

Candidate background:
{context}

Job Description:
{jd_text[:1500]}

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

        raw = _call_llm(prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            result = json.loads(raw)
        except Exception:
            result = {
                "JOB_TITLE": parsed.get("JOB_TITLE", "Unknown"),
                "COMPANY": parsed.get("COMPANY", "Unknown"),
                "match_pct": 50,
                "top_skills": [],
                "top_gaps": [],
                "difficulty": "Medium",
                "recommendation": "Could not parse comparison result.",
            }
        results.append(result)

    return results

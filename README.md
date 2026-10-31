# JobGenie RAG Assistant

An AI-powered career co-pilot that analyzes your resume against any job description, generating skill gap reports, 60-second pitches, interview question predictions, and confidence scores in real time.

[![Live Demo](https://img.shields.io/badge/Live_Demo-HuggingFace-orange?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/Utkarsh94123/jobgenie-rag-assistant)
[![CI](https://github.com/utkarshkapoor95/jobgenie-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/utkarshkapoor95/jobgenie-rag-assistant/actions)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)

---

## Why This Exists

Job seekers spend hours manually comparing their resume to job descriptions and still miss critical skill gaps, fail to articulate their value clearly, or walk into interviews underprepared.

JobGenie solves this with a **Retrieval-Augmented Generation (RAG)** pipeline that ingests your resume and any JD, then produces actionable career intelligence in seconds. Built specifically for candidates transitioning across domains, where framing transferable skills correctly can be the difference between a screening call and a rejection.

---

## RAGAS Evaluation Results

All outputs are evaluated using the [RAGAS framework](https://docs.ragas.io/) across 20 test queries spanning skill gap analysis, pitch generation, and interview prediction.

| Metric | Score | What It Measures |
|---|---|---|
| **Faithfulness** | **0.87** | Are generated answers grounded in retrieved resume context? |
| **Answer Relevancy** | **0.82** | Does the output actually address the user's query? |
| **Context Precision** | **0.79** | Are the retrieved chunks relevant to the question asked? |

> Evaluation script: [`ragas_eval.py`](ragas_eval.py) | Results: [`eval_results/ragas_scores.json`](eval_results/ragas_scores.json)

Scores computed on `llama-3.3-70b-versatile` via Groq, with `all-MiniLM-L6-v2` embeddings and Pinecone vector retrieval (top-5).

---

## Live Demo

**[Try it on HuggingFace Spaces](https://huggingface.co/spaces/Utkarsh94123/jobgenie-rag-assistant)**

Upload your resume (PDF), paste any job description, and get your full career intelligence report in under 30 seconds.

---

## Features

| Feature | What It Does |
|---|---|
| **Find Matching Jobs** | Live job search via Adzuna (19 countries). Scores every result against your resume, shows a fit percentage per job, apply links, and aggregates the skills most commonly missing across all results, with learning links for each |
| **Skill Gap Analysis** | Compares your resume against a pasted JD. Highlights matches, gaps, and how to reframe transferable experience |
| **60-Second Pitch** | Generates a personalized elevator pitch tailored to the specific role and company |
| **Interview Predictor** | Predicts 10 likely interview questions (technical, behavioral, situational) with answer frameworks |
| **Multi-JD Comparison** | Side-by-side comparison of up to 3 JDs with match percentages and a best-fit recommendation |
| **Confidence Score** | Numeric readiness score broken down by technical skills, experience, education, and portfolio match |
| **Q&A Mode** | Free-form questions about your profile: salary expectations, career switch framing, skill positioning |

### Find Matching Jobs: how it works

1. Upload your resume
2. Answer four questions: job title, country, city, salary expectation
3. The app pulls live listings from Adzuna and scores each one against your resume (fit %, matched skills, missing skills)
4. Results are ranked by fit, each with a direct apply link
5. Across all returned jobs, the app aggregates which skills come up missing most often ("missing in 6 of 10 listings") and generates a learning-search link (YouTube, freeCodeCamp, Coursera) for each — reverse-engineering what the market is actually asking for, not just one job at a time

---

## Architecture

```
                        +------------------+
                        |   User uploads   |
                        |  Resume (PDF) +  |
                        |  Job Description |
                        +--------+---------+
                                 |
                    +------------v-----------+
                    |   Chainlit Chat UI     |
                    |   (session, streaming)  |
                    +------------+-----------+
                                 |
              +------------------v------------------+
              |         Document Processing         |
              |  PyMuPDF/PyPDF2 -> Text Extraction  |
              +------------------+------------------+
                                 |
                    +------------v-----------+
                    |    Text Chunking       |
                    |  RecursiveCharacter    |
                    |  (400 chars, 60 overlap)|
                    +------------+-----------+
                                 |
              +------------------v------------------+
              |         Embedding Layer             |
              |  all-MiniLM-L6-v2 (384-dim vectors) |
              +------------------+------------------+
                                 |
                    +------------v-----------+
                    |   Pinecone Vector DB   |
                    |   (Serverless, cosine) |
                    +------------+-----------+
                                 |
              +------------------v------------------+
              |         RAG Chain (LangChain)        |
              |  Semantic retrieval (top-5 chunks)   |
              |  + Prompt templates per feature      |
              +------------------+------------------+
                                 |
                    +------------v-----------+
                    |   Groq LPU Inference   |
                    |  LLaMA 3.3 70B (fast)  |
                    +------------+-----------+
                                 |
              +------------------v------------------+
              |            Output Layer             |
              |  Gap Analysis | Pitch | Questions   |
              |  Comparison Table | Confidence Score|
              +-------------------------------------+
```

---

## Tech Stack

| Layer | Technology | Why This Choice |
|---|---|---|
| **LLM** | Groq (LLaMA 3.3 70B) | Ultra-low latency inference via LPU hardware, free tier |
| **RAG Framework** | LangChain | Document loaders, text splitters, prompt templates, chain composition |
| **Vector Store** | Pinecone (Serverless) | Managed vector DB with cosine similarity, free Starter tier |
| **Embeddings** | all-MiniLM-L6-v2 | Lightweight 384-dim sentence embeddings, fast on CPU |
| **Evaluation** | RAGAS | Industry-standard RAG evaluation (faithfulness, relevancy, precision) |
| **Chat UI** | Chainlit | Streaming responses, session management, PDF upload, action buttons |
| **Containerization** | Docker + Compose | Reproducible builds, one-command deployment |
| **CI/CD** | GitHub Actions | Automated linting (Ruff) + Docker build verification on every push |
| **Deployment** | HuggingFace Spaces | Free Docker-based hosting with public URL |

---

## Project Structure

```
jobgenie-rag-assistant/
├── .github/
│   └── workflows/
│       └── ci.yml              # Ruff lint + Docker build on push
├── knowledge_base/             # Static knowledge base files (optional)
├── eval_results/
│   └── ragas_scores.json       # RAGAS evaluation output
├── app.py                      # Chainlit entry point + UI logic
├── rag_pipeline.py             # Core RAG chain: retrieval, prompts, LLM calls
├── embeddings.py               # Pinecone indexing + embedding utilities
├── jd_parser.py                # Job description structured extraction
├── ragas_eval.py               # RAGAS evaluation script (reproducible)
├── Dockerfile                  # Python 3.11, non-root user, port 7860
├── docker-compose.yml          # One-command local dev
├── requirements.txt            # Pinned dependencies
├── .env.example                # Required environment variables
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+ (3.11 recommended)
- Docker (optional but recommended)
- [Groq API key](https://console.groq.com/) (free, no credit card)
- [Pinecone API key](https://app.pinecone.io/) (free Starter tier, no credit card)

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/utkarshkapoor95/jobgenie-rag-assistant.git
cd jobgenie-rag-assistant

cp .env.example .env
# Add your GROQ_API_KEY and PINECONE_API_KEY to .env

docker-compose up --build
```

App runs at `http://localhost:7860`

### Option 2: Local Python

```bash
git clone https://github.com/utkarshkapoor95/jobgenie-rag-assistant.git
cd jobgenie-rag-assistant

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Add your GROQ_API_KEY and PINECONE_API_KEY to .env

chainlit run app.py
```

### Run RAGAS Evaluation

```bash
python ragas_eval.py
# Results saved to eval_results/ragas_scores.json
```

---

## Environment Variables

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=jobgenie-index
HF_TOKEN=your_huggingface_token    # Only needed for HuggingFace Spaces deployment
ADZUNA_APP_ID=your_adzuna_app_id   # Required for Find Matching Jobs
ADZUNA_APP_KEY=your_adzuna_app_key # Required for Find Matching Jobs
```

All APIs used are **free tier** with no credit card required.

| Service | Free Tier Limit | Enough For This Project? |
|---|---|---|
| Groq (`openai/gpt-oss-120b`) | 30 req/min, 1,000 req/day, 200K tokens/day | Yes (portfolio demo) |
| Pinecone | 2 GB storage, 1M reads/month, 5 indexes | Yes (resume chunks are tiny) |
| Adzuna | ~1,000 calls/month, 25 req/min | Yes (one search = one call per session) |
| HuggingFace Spaces | Free CPU-basic Docker hosting | Yes (sleeps when idle) |

> **Note:** Groq retired `llama-3.3-70b-versatile` on August 16, 2026. This project uses `openai/gpt-oss-120b`, Groq's recommended replacement with the same free-tier shape and a higher daily token cap.

---

## Sample Output

**Input:** Senior Data Analyst JD + MBA student resume with operations background

### Confidence Score

| Category | Score |
|---|---|
| Technical Skills Match | 65% |
| Experience Relevance | 78% |
| Education Fit | 85% |
| Project Portfolio Match | 70% |
| **Overall Confidence** | **72%** |

### Generated Pitch (Preview)

> "I bring 3+ years of operations experience where I translated messy ground-level data into decisions that reduced costs and improved process efficiency. Now, with an MBA and hands-on AI projects including this RAG assistant and a multi-agent startup launch kit, I combine business context with technical execution. I don't just build dashboards; I build ones that actually get used."

---

## CI/CD

Every push to `main` triggers:

1. **Ruff Linter** - Python code quality and style checks
2. **Dependency Verification** - Confirms all imports resolve
3. **Docker Build** - Validates the container builds successfully

[![CI Status](https://github.com/utkarshkapoor95/jobgenie-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/utkarshkapoor95/jobgenie-rag-assistant/actions)

---

## Built By

**Utkarsh Kapoor** | AI & Marketing Automation

MBA candidate building intelligent business workflows using n8n, LLM APIs, and Python. This is one of six portfolio projects demonstrating RAG, multi-agent orchestration, and production automation skills.

[![Portfolio](https://img.shields.io/badge/Portfolio-autothynk.com-blue?style=flat)](https://autothynk.com)
[![GitHub](https://img.shields.io/badge/GitHub-utkarshkapoor95-181717?style=flat&logo=github)](https://github.com/utkarshkapoor95)

---

## License

MIT License

---
title: JobGenie RAG Assistant
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# 🎯 JobGenie RAG Assistant

A RAG-powered career assistant that analyzes your resume, recommends matching jobs via live search, scores your fit with skill gap analysis, and generates interview prep materials.

**[Try the Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/Utkarsh94123/jobgenie-rag-assistant)**

---

## What It Does

Upload your resume PDF and JobGenie:

1. **Embeds your resume** into a Pinecone vector database for semantic retrieval
2. **Suggests 3 job titles** by analyzing your resume with an LLM (one-click selection)
3. **Searches live job listings** across 6 countries via JSearch API (Google for Jobs index)
4. **Scores each job** against your resume: fit percentage, matched skills, missing skills
5. **Aggregates skill gaps** across all results with learning resource links (YouTube, freeCodeCamp, Coursera)
6. **Analyzes individual JDs**: skill gap analysis, 60-second pitch, interview question prediction, readiness score
7. **Compares up to 3 JDs** side-by-side with a recommendation

---

## Architecture

```
Resume PDF
    |
    v
[PyMuPDF / pypdf] --> Extract text
    |
    v
[LangChain TextSplitter] --> Chunk (400 chars, 60 overlap)
    |
    v
[all-MiniLM-L6-v2] --> Embed chunks
    |
    v
[Pinecone Serverless] --> Store & retrieve (cosine similarity)
    |
    v
[Groq LLM: openai/gpt-oss-120b] --> Suggest titles / Score jobs / Analyze gaps
    |
    v
[JSearch API] --> Live job listings (200 req/month free tier)
    |
    v
[Chainlit UI] --> Interactive chat with buttons + results
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Chat UI | Chainlit |
| RAG Framework | LangChain |
| Vector Database | Pinecone (Serverless, AWS us-east-1) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384 dim) |
| LLM | Groq (openai/gpt-oss-120b) |
| Job Search API | JSearch by OpenWeb Ninja (free, 200 req/month) |
| RAG Evaluation | RAGAS |
| Containerization | Docker |
| CI/CD | GitHub Actions (Ruff lint + dependency check + Docker build) |
| Deployment | HuggingFace Spaces (Docker SDK) |

---

## RAGAS Evaluation Scores

Evaluated on 20 test queries covering skill matching, gap analysis, pitch generation, interview prep, and career advice:

| Metric | Score |
|---|---|
| Faithfulness | **0.87** |
| Answer Relevancy | **0.82** |
| Context Precision | **0.79** |

Full evaluation data: [`eval_results/ragas_scores.json`](eval_results/ragas_scores.json)

---

## Project Structure

```
jobgenie-rag-assistant/
├── app.py                  # Chainlit UI and conversation flow
├── config.py               # Centralized constants, prompts, model config
├── rag_pipeline.py         # RAG: embedding, retrieval, LLM calls, analysis
├── job_search.py           # JSearch API integration, scoring, skill aggregation
├── ragas_eval.py           # RAGAS evaluation script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container config for HuggingFace Spaces
├── docker-compose.yml      # Local Docker development
├── .env.example            # Required environment variables
├── .github/workflows/ci.yml  # CI: lint + imports + Docker build
├── eval_results/
│   └── ragas_scores.json   # RAGAS evaluation results
└── docs/
    └── index.html          # Project landing page
```

---

## Key Design Decisions

**Why JSearch over Adzuna?** Adzuna's free tier is only 25 test calls total, then requires a paid subscription with a credit card. JSearch is genuinely free: 200 requests/month, no credit card required.

**Why no salary filter?** Most job postings (especially in India) don't include salary data. Adding a salary question creates friction without reliable payoff. Confirmed via JSearch's own FAQ and live testing.

**Why AI-suggested titles?** Eliminates the cold-start problem. Instead of asking users to type a job title (which many don't know how to phrase for optimal search results), the LLM analyzes their resume and suggests 3 high-volume titles. One click to proceed.

**Why Pinecone over FAISS/Chroma?** Pinecone's serverless tier is free, persists across sessions, and requires zero infrastructure. FAISS would need in-memory storage that resets on every Space restart. Chroma would need persistent disk (not available on HuggingFace free tier).

**Why openai/gpt-oss-120b?** The previous model (llama-3.3-70b-versatile) was retired by Groq on 2026-08-16. gpt-oss-120b is Groq's recommended replacement with the same free tier limits (30 RPM, 1,000 req/day) and a higher daily token cap.

---

## Local Development

```bash
# Clone
git clone https://github.com/utkarshkapoor95/jobgenie-rag-assistant.git
cd jobgenie-rag-assistant

# Set up environment
cp .env.example .env
# Fill in your API keys in .env

# Install dependencies
pip install -r requirements.txt

# Run
chainlit run app.py --port 7860
```

Or with Docker:

```bash
docker-compose up --build
```

---

## Environment Variables

| Variable | Required | Where to Get |
|---|---|---|
| `GROQ_API_KEY` | Yes | [console.groq.com](https://console.groq.com/) (free) |
| `PINECONE_API_KEY` | Yes | [app.pinecone.io](https://app.pinecone.io/) (free) |
| `JSEARCH_API_KEY` | Yes | [app.openwebninja.com](https://app.openwebninja.com/signup) (free, no CC) |
| `HF_TOKEN` | Spaces only | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## Known Limitations

- **JSearch coverage**: Niche job titles (e.g. "AI Automation Orchestrator") may return zero results because Google's index doesn't have enough matching listings. Broader titles work better.
- **Salary data**: Spotty/unparsed free text per multiple independent sources. This is a JSearch/Google for Jobs limitation, not a bug.
- **HuggingFace free tier**: Space sleeps after ~48 hours of no traffic. First visit after sleep takes 30-60 seconds to wake up.
- **Groq free tier**: 30 RPM, 1,000 requests/day. Sufficient for demo use, not production scale.

---

## License

MIT

---

Built by [Utkarsh Kapoor](https://github.com/utkarshkapoor95)

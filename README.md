# 🧞 JobGenie RAG Assistant

> An AI-powered career co-pilot that analyzes your resume against any job description — generating skill gap reports, 60-second pitches, interview question predictions, and confidence scores in real time.

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-orange?style=for-the-badge)](https://huggingface.co/spaces/Utkarsh94123/jobgenie-rag-assistant)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge&logo=chainlink)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-F55036?style=for-the-badge)](https://groq.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-00BCD4?style=for-the-badge)](https://www.pinecone.io/)
[![Chainlit](https://img.shields.io/badge/Chainlit-Chat%20UI-FF4B4B?style=for-the-badge)](https://chainlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)

---

## 🎯 Problem Statement

Job seekers spend hours manually comparing their resume to job descriptions — and often still miss critical skill gaps, fail to articulate their value clearly, or walk into interviews underprepared.

JobGenie solves this with a Retrieval-Augmented Generation (RAG) pipeline that ingests your resume and any JD, then produces actionable career intelligence in seconds. Built for candidates transitioning across domains — where framing your transferable skills correctly can be the difference between a screening call and a rejection.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Skill Gap Analysis** | Compares your resume against the JD and highlights missing skills, partial matches, and strong overlaps |
| 🎤 **60-Second Pitch Generator** | Creates a personalized, interview-ready elevator pitch tailored to the specific role |
| ❓ **Interview Question Predictor** | Predicts the top 5–8 likely interview questions based on the JD and your background |
| 📊 **JD Comparison Table** | Side-by-side structured comparison of JD requirements vs. your profile |
| 💡 **Confidence Score Dashboard** | Numeric readiness score with breakdown by category (technical, domain, soft skills) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq · LLaMA 3.3 70B (ultra-low latency inference) |
| **RAG Framework** | LangChain (document loaders, chains, prompt templates) |
| **Vector Store** | Pinecone (semantic search over resume + JD embeddings) |
| **Chat UI** | Chainlit (streaming, session management, file upload) |
| **Containerization** | Docker + Docker Compose |
| **Deployment** | HuggingFace Spaces |
| **Embeddings** | HuggingFace Sentence Transformers |

---

## 🏗️ Architecture

```
User Input (Resume PDF + Job Description)
        │
        ▼
  Document Loader (LangChain)
        │
        ▼
  Text Chunking & Embedding (HuggingFace)
        │
        ▼
  Pinecone Vector Store ◄──── Semantic Retrieval
        │
        ▼
  RAG Chain (LangChain + Groq LLaMA 3.3 70B)
        │
        ▼
  ┌─────────────────────────────────────┐
  │  Skill Gap │ Pitch │ Questions │ Score │
  └─────────────────────────────────────┘
        │
        ▼
  Chainlit Chat Interface
```

---

## 🚀 Live Demo

👉 **[Try it on HuggingFace Spaces](https://huggingface.co/spaces/Utkarsh94123/jobgenie-rag-assistant)**

Upload your resume (PDF), paste any job description, and get your full career intelligence report in under 30 seconds.

---

## 📁 Project Structure

```
jobgenie-rag-assistant/
├── app.py                  # Chainlit app entry point
├── rag_pipeline.py         # Core RAG chain (LangChain + Groq)
├── vector_store.py         # Pinecone indexing & retrieval
├── prompts/
│   ├── skill_gap.py        # Skill gap analysis prompt template
│   ├── pitch_generator.py  # 60-sec pitch prompt template
│   ├── interview_qs.py     # Interview question predictor
│   └── confidence.py       # Confidence score prompt
├── utils/
│   ├── pdf_parser.py       # Resume PDF extraction
│   └── jd_parser.py        # Job description preprocessing
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.10+
- Docker (optional but recommended)
- Pinecone API key
- Groq API key

### Option 1: Run with Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/jobgenie-rag-assistant.git
cd jobgenie-rag-assistant

# Set environment variables
cp .env.example .env
# Fill in your PINECONE_API_KEY and GROQ_API_KEY in .env

# Build and run
docker-compose up --build
```

App will be live at `http://localhost:8000`

### Option 2: Run Locally

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/jobgenie-rag-assistant.git
cd jobgenie-rag-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PINECONE_API_KEY=your_key_here
export GROQ_API_KEY=your_key_here

# Run the app
chainlit run app.py
```

---

## 🔑 Environment Variables

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=jobgenie-index
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
HF_TOKEN=your_huggingface_token   # Only needed for HF Spaces deployment
```

---

## 📈 Sample Output

**Input:** Senior Data Analyst JD + MBA student resume with ops background

**Confidence Score:** `72/100`

| Category | Score | Gap |
|---|---|---|
| Technical Skills | 65/100 | Missing: dbt, Airflow |
| Domain Knowledge | 85/100 | Strong: operations analytics |
| Communication | 90/100 | Strong: MBA + stakeholder exp. |
| AI/ML Readiness | 60/100 | Improving: RAG, LangChain |

**Generated Pitch Preview:**
> *"I bring 3+ years of operations experience where I translated messy ground-level data into decisions that reduced costs and improved process efficiency. Now, with an MBA and hands-on AI projects like this RAG assistant, I combine business context with technical execution — which means I don't just build dashboards, I build ones that actually get used."*

---

## 🔮 Roadmap

- [ ] LinkedIn profile ingestion (in addition to PDF resume)
- [ ] Multi-JD batch comparison mode
- [ ] ATS score simulator
- [ ] Resume rewrite suggestions based on skill gaps
- [ ] Shareable PDF report export

---

## 🙋 About the Builder

Built by **Utkarsh Kapoor** — MBA candidate transitioning from Operations to AI/Data roles.

This project reflects my belief that the best data tools solve real human problems — and nothing is more stressful than a job search. JobGenie is my attempt to give every candidate access to the kind of career intelligence that was previously locked behind expensive coaches.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Profile-orange?style=flat)](https://huggingface.co/Utkarsh94123)

---

## 📄 License

MIT License — free to use, fork, and build upon.

---

*If this helped you, drop a ⭐ — it means a lot and helps others find the project.*

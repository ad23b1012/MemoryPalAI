# 🧠 MemoryPalAI — Intelligent Knowledge Workspace (v2.0)
### Next-Gen Agentic AI Tutor • Style-Aware Grounded RAG • Adaptive Quiz Arena • Spaced Repetition Companion

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Google-Gemini_Flash-4285F4?logo=google&logoColor=white)](https://ai.google.dev)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000?logo=pinecone&logoColor=white)](https://pinecone.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent_Fallback-orange)](https://trychroma.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Overview

**MemoryPalAI** is a production-grade Agentic AI Knowledge Workspace and Personal Learning Companion. It transforms unstructured study materials—such as academic PDFs, lecture audio, DOCX notes, and web articles—into an interactive, grounded learning ecosystem.

MemoryPalAI autonomously:
1. **Ingests & Analyzes:** Extracts text, detects pedagogical style, tone, topics, and conceptual tags.
2. **Embeds & Indexes:** Uses Gemini `text-embedding-004` with a zero-crash **Unified Vector Store** (Pinecone cloud with automatic persistent ChromaDB fallback).
3. **Retrieves & Grounds:** Delivers strictly grounded answers with citations or general knowledge fallback.
4. **Architects Roadmaps:** Synthesizes customized 3-phase mastery study plans with curated YouTube resources.
5. **Generates Quizzes:** Designs conceptual multiple-choice questions (MCQs) and evaluates student answers.
6. **Auto-Revises:** Detects weak concepts (<70% score) to synthesize concise revision packs and active recall flashcards.
7. **Schedules Spaced Repetition:** Calculates SuperMemo SM-2 review intervals and tracks mastery memory across topics.
8. **Visualizes Knowledge:** Maps entities and relationships into an interactive physics-driven concept graph.

---

## 🏗️ System Architecture

```
                  ┌──────────────────────────────────────────────────┐
                  │                   User Inputs                    │
                  │   (PDF / DOCX / TXT / Audio MP3-WAV / Web URLs)  │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │                 Ingestion Agent                  │
                  │   Whisper Transcription • PyMuPDF • PyPDF        │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │          Style & Concept Organizer Agent         │
                  │  Style Detection (Explanatory/Technical/Example) │
                  │  Tone (Formal/Friendly) • Recursive Chunker      │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │               Embedding Service                  │
                  │        Google Gemini text-embedding-004          │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │              Unified Vector Store                │
                  │   Pinecone (Serverless) ⇄ ChromaDB (Persistent)  │
                  └─────────────────────────┬────────────────────────┘
                                            │
         ┌──────────────────────────────────┼─────────────────────────────────┐
         ▼                                  ▼                                 ▼
┌──────────────────┐              ┌──────────────────┐              ┌──────────────────┐
│  Grounded RAG QA │              │ Interactive Graph│              │ Adaptive Quiz &  │
│  & Study Roadmap │              │ (PyVis Physics)  │              │  Revision Deck   │
└──────────────────┘              └──────────────────┘              └────────┬─────────┘
                                                                             │
                                                                             ▼
                                                                    ┌──────────────────┐
                                                                    │ Spaced Repetition│
                                                                    │  (SM-2 Tracker)  │
                                                                    └──────────────────┘
```

---

## 🌟 Key Features & Improvements (v2.0)

| Module | Features & Capabilities |
| :--- | :--- |
| **🎨 Futuristic Glassmorphic UI** | Responsive dark glassmorphism theme with gradient telemetry badges, animated metric cards, interactive radio choices, and clean tab navigation. |
| **🛡️ Zero-Crash Architecture** | Eliminates fatal top-level exits. Gracefully falls back to ChromaDB if Pinecone keys are missing, and to SQLite if PostgreSQL is absent. |
| **📥 Multimodal Ingestion** | Ingests PDF (PyMuPDF), DOCX (python-docx), TXT, Markdown, Web URLs, and audio lectures (MP3, WAV, M4A) with on-demand Whisper transcription and preview audio player. |
| **🔍 Transparent Grounded RAG** | Inspect retrieved chunks, similarity metrics, and source document metadata directly in the conversation drawer. |
| **🗺️ 3-Phase Mastery Roadmap** | Generates Foundations, Deep Dive, and Advanced Mastery milestones with direct YouTube tutorial search links. |
| **🎯 Adaptive Quiz Arena** | MCQ quizzes with customized question lengths, formatted option selections, and detailed pedagogical evaluations. |
| **🔁 Smart Auto-Revision** | Automatically triggers when a quiz score is below 70%, synthesizing key takeaways, target video lessons, and self-test flashcards. |
| **📈 Spaced Repetition Tracker** | SuperMemo SM-2 algorithm calculates next review intervals (1, 2, 5, 10+ days) and alerts you on due topics. |
| **🕸️ Interactive Concept Graph** | Physics-based interactive concept network with color-coded nodes for Topics, Concepts, Phrases, and Tags. |
| **🐳 Production Server Ready** | Optimized Dockerfile with dynamic `$PORT` support, docker-compose production bundle with Nginx reverse proxy, and WebSocket stream handling. |

---

## ⚙️ Environment Configuration

Copy `.env.example` to `.env` and configure your credentials:

```bash
cp .env.example .env
```

| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `GOOGLE_API_KEY` | **Required for AI** — Google Gemini API key ([Get Key](https://aistudio.google.com/)) | `AIzaSy...` |
| `GEMINI_MODEL` | Gemini LLM model | `gemini-1.5-flash` *(supports `gemini-2.0-flash`, `gemini-1.5-pro`)* |
| `PINECONE_API_KEY` | Pinecone Cloud API Key *(optional; falls back to ChromaDB)* | `pcsk_...` |
| `PINECONE_INDEX_NAME` | Pinecone index name | `memorypalai` |
| `PINECONE_NAMESPACE` | Pinecone namespace | `memorypal` |
| `VECTOR_BACKEND` | `auto`, `pinecone`, or `chroma` | `auto` |
| `POSTGRES_USER` | PostgreSQL user *(optional; falls back to SQLite)* | `memorypal_user` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `memorypal_pass` |
| `POSTGRES_DB` | PostgreSQL database name | `memorypal` |
| `POSTGRES_HOST` | PostgreSQL host | `postgres_db` *(or `localhost`)* |
| `PORT` | Web server port | `8501` *(or cloud `$PORT`)* |

---

## 🚀 Quickstart Guide

### 1. Local Python Setup

```bash
# Clone the repository
git clone https://github.com/ad23b1012/MemoryPalAI.git
cd MemoryPalAI

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run system verification test
python test.py

# Launch Streamlit app
streamlit run frontend/streamlit_app.py
```

Open your browser at `http://localhost:8501`.

---

### 2. Docker & Docker Compose (Recommended for Production)

#### Local Development with Docker Compose:
```bash
docker-compose up --build
```
This starts:
- MemoryPalAI App at `http://localhost:8501`
- PostgreSQL 16 at `localhost:5432`

#### Full Production Stack (with Nginx Reverse Proxy):
```bash
docker-compose -f docker-compose.prod.yml up -d --build
```
This runs:
- **Nginx Reverse Proxy** at `http://localhost:80` (with WebSocket upgrade, 100MB upload limit, gzip compression)
- **MemoryPalAI App Container**
- **PostgreSQL 16 Database**

To check logs or status:
```bash
docker-compose -f docker-compose.prod.yml logs -f
```

---

### 3. Deploying to Cloud Servers

#### Deploy to Render:
1. Connect this GitHub repository in [Render Dashboard](https://dashboard.render.com).
2. Choose **Blueprint** deployment: Render will automatically detect `render.yaml`.
3. Fill in your `GOOGLE_API_KEY` (and optional `PINECONE_API_KEY`) under Environment Variables.
4. Render deploys the app and managed PostgreSQL automatically!

#### Deploy to Railway / Fly.io / Cloud Run / VPS:
MemoryPalAI includes native `$PORT` binding in its `Dockerfile` and `Procfile`.
- **Railway:** Simply connect GitHub repo; Railway auto-detects `Dockerfile` and binds `$PORT`.
- **VPS (Ubuntu/Debian):** Install Docker & Docker Compose, clone repository, run `docker-compose -f docker-compose.prod.yml up -d`.

---

## 🧪 Verification & Testing

Verify that all backend services, vector database, and AI pipelines are working:

```bash
python test.py
```

Expected output:
```
============================================================
🧠 MemoryPalAI - System Health & Integration Check
============================================================
✅ Configuration loaded: MemoryPalAI v2.0.0
   - Target Gemini Model: gemini-1.5-flash
   - Target Vector Backend: auto
   - Database URL Type: SQLite (Local Fallback)
✅ Vector Store initialized successfully (Active Backend: chroma)
✅ Relational Database tables verified/created successfully.
✅ All Core Agents initialized successfully:
   - IngestionAgent, QuizAgent, RevisionAgent, PlannerAgent, OrganizerAgent
   - Spaced Repetition Engine: test interval = 5 days (Progressing Well)
============================================================
🎉 All systems operational! App is ready for local use and server deployment.
============================================================
```

---

## 👥 Contributors & License

- **Author:** Abhishek Buddiga & Izhaar Ahmed
- **License:** MIT License. See [LICENSE](LICENSE) for details.

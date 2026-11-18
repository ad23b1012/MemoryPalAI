# 🧠 MemoryPalAI — Autonomous Cognitive Learning OS (v2.0)
### Agentic AI Multi-Agent Architecture • Grounded Citation RAG • Adaptive Quiz Arena • SuperMemo SM-2 Spaced Repetition

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js 15](https://img.shields.io/badge/Next.js-15.1_App_Router-black?logo=next.js&logoColor=white)](https://nextjs.org)
[![React 19](https://img.shields.io/badge/React-19.0-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Gemini 3.8 Flash](https://img.shields.io/badge/Google-Gemini_3.8_Flash-4285F4?logo=google&logoColor=white)](https://ai.google.dev)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_Database-000000?logo=pinecone&logoColor=white)](https://pinecone.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supabase_Enterprise-4169E1?logo=postgresql&logoColor=white)](https://supabase.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌐 Live Production Deployments

* **Backend Core Engine:** [https://memorypal-app.onrender.com](https://memorypal-app.onrender.com) *(Mirror: [https://memorypalai-1-71yi.onrender.com](https://memorypalai-1-71yi.onrender.com))*
* **Interactive OpenAPI Docs:** [https://memorypal-app.onrender.com/docs](https://memorypal-app.onrender.com/docs)
* **System Telemetry & Health:** [https://memorypal-app.onrender.com/api/health](https://memorypal-app.onrender.com/api/health)
* **Frontend Web Application:** Deployed via Vercel with edge-rewrites to Render backend.

---

## 🚀 Overview

**MemoryPalAI** is a production-grade Agentic AI Knowledge Workspace and Autonomous Learning Companion. It transforms unstructured study materials—such as academic PDFs, lecture recordings, research notes, and web publications—into an interconnected, verifiable cognitive learning ecosystem.

The system orchestrates specialized autonomous agents:
1. **Multimodal Ingestion Agent:** Ingests PDFs (PyMuPDF/pypdf), Word documents (python-docx), plain text/Markdown, web articles (BeautifulSoup), and audio lectures (Whisper MP3/WAV/M4A).
2. **Pedagogical Style & Taxonomy Agent:** Analyzes pedagogical tone, text structure, topic domains, and semantic key concepts.
3. **High-Dimensional Vector Indexer:** Generates 768-dimensional multimodal embeddings using Google `gemini-embedding-2` and persists them in Pinecone Cloud with local ChromaDB fallback.
4. **Grounded Synthesis & Citation Engine:** Synthesizes contextual answers strictly bounded by retrieved source chunks using Google `gemini-3.8-flash` with dual-key automatic failover.
5. **Study Roadmap Planner:** Generates customized 3-phase mastery study plans with targeted conceptual milestones and curated YouTube technical resources.
6. **Adaptive Quiz Arena:** Synthesizes document-grounded multiple-choice assessments, evaluates user answers with exact mathematical scoring, and provides step-by-step conceptual explanations.
7. **Spaced Repetition Engine:** Implements the **SuperMemo SM-2** exponential decay algorithm to compute optimal review intervals (1, 2, 5, 10+ days) and track mastery retention over time.
8. **Concept Graph Service:** Generates hierarchical entity-relationship graphs mapping subjects, concepts, and semantic tags.

---

## 🏗️ System Architecture

```
                          ┌────────────────────────────────────────────────────────┐
                          │                      Client Tier                       │
                          │   Next.js 15 (App Router, React 19, TypeScript)        │
                          └───────────────────────────┬────────────────────────────┘
                                                      │  /api/* (Reverse Proxy / CORS)
                                                      ▼
                          ┌────────────────────────────────────────────────────────┐
                          │               FastAPI Asynchronous Core                │
                          │      Uvicorn ASGI • Pydantic v2 • Connection Pool      │
                          └───────────────────────────┬────────────────────────────┘
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         ▼                                            ▼                                            ▼
┌──────────────────┐                       ┌──────────────────┐                       ┌──────────────────┐
│ Ingestion Agent  │                       │  LLM Service     │                       │ Unified Vector   │
│ Whisper Audio    │                       │ Gemini 3.8 Flash │                       │ Store & Relational│
│ PyMuPDF / docx   │                       │ Multi-Key Fallover│                      │ Pinecone Cloud   │
│ BeautifulSoup    │                       │ gemini-embed-2   │                       │ Supabase Postgres│
└────────┬─────────┘                       └────────┬─────────┘                       └────────┬─────────┘
         │                                          │                                          │
         └──────────────────────────────────────────┼──────────────────────────────────────────┘
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
        ┌───────────────────────┐                                         ┌───────────────────────┐
        │  Cognitive Tutor &    │                                         │   Adaptive Quiz &     │
        │  Grounded Citations   │                                         │   SuperMemo SM-2      │
        └───────────────────────┘                                         └───────────────────────┘
```

---

## 🌟 Core Technical Capabilities

| Module | Technical Implementation |
| :--- | :--- |
| **⚡ Next.js 15 App Router** | Full-stack TypeScript architecture on React 19 with modular client components and centralized `/lib/api.ts` typed fetch client. |
| **🚀 FastAPI Asynchronous Engine** | High-throughput async REST endpoints with strict Pydantic v2 schema validation, CORS security headers, and automated OpenAPI documentation. |
| **🧠 Gemini 3.8 Flash Core** | Uses the official `google.genai` SDK with automatic quota and rate-limit multi-key failover (`GOOGLE_API_KEY` ➔ `GOOGLE_API_KEY_FALLBACK`). |
| **📐 Gemini Embedding 2** | Generates 768-dimensional dense vector embeddings with Matryoshka Representation Learning (MRL). |
| **🌲 Distributed Vector Network** | Pinecone Serverless cloud indexing with cosine similarity scoring, namespace isolation, and graceful fallback to persistent local ChromaDB. |
| **🐘 Enterprise Relational Core** | PostgreSQL persistence via Supabase connection pooling with fallback to SQLite for local development. |
| **🎯 Document-Grounded Quiz Arena** | Dynamically extracts chunks from ingested documents to synthesize questions testing specific course materials, calculating true percentage scores. |
| **📈 SuperMemo SM-2 Spaced Repetition** | Computes exponential memory retention curves, Easiness Factors ($EF \ge 1.3$), repetition counts, and scheduled review dates. |
| **🕸️ Knowledge Graph Extraction** | Entity-relationship graph generation mapping document subjects, extracted concepts, and semantic connections. |
| **🐳 Containerized Orchestration** | Multi-stage production `Dockerfile` with dynamic `$PORT` binding, pre-configured for Render, Railway, Fly.io, and Docker Compose. |

---

## ⚙️ Environment Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

| Environment Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `GOOGLE_API_KEY` | Primary Google Gemini API Key | `AIzaSy...` |
| `GOOGLE_API_KEY_FALLBACK` | Secondary Gemini API Key for automatic failover | `AIzaSy...` |
| `GEMINI_MODEL` | Gemini LLM Model | `gemini-3.8-flash` |
| `EMBEDDING_MODEL` | Text & Multimodal Embedding Model | `gemini-embedding-2` |
| `PINECONE_API_KEY` | Pinecone Cloud Vector Store API Key | `pcsk_...` |
| `PINECONE_INDEX_NAME` | Target Pinecone Vector Index | `memorypalai` |
| `PINECONE_NAMESPACE` | Vector namespace partition | `memorypal` |
| `VECTOR_BACKEND` | Vector storage provider (`auto`, `pinecone`, `chroma`) | `auto` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:...@host:5432/postgres` |
| `PORT` | Local FastAPI server port | `8501` |
| `NEXT_PUBLIC_BACKEND_URL` | Target backend URL for Next.js API proxy | `https://memorypal-app.onrender.com` |

---

## 🚀 Local Development Setup

### 1. Prerequisites
* **Python 3.11+**
* **Node.js 20+** and **npm 10+**

### 2. Universal One-Command Startup
The project includes a root startup script that concurrently runs both FastAPI and Next.js:

```bash
# Clone the repository
git clone https://github.com/ad23b1012/MemoryPalAI.git
cd MemoryPalAI

# Run universal startup script
./start.sh
```

---

### 3. Manual Component Startup

#### Backend Service (FastAPI):
```bash
# Setup Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python3 -m app.server
```
*Backend active at: `http://localhost:8501`*

#### Frontend Service (Next.js 15):
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Next.js development server
npm run dev
```
*Frontend active at: `http://localhost:3000`*

---

## 📡 Core API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System telemetry, vector index metrics, and engine status |
| `GET` | `/api/documents` | List indexed documents, topic clusters, and chunk counts |
| `POST` | `/api/ingest` | Multipart upload (PDF, TXT, DOCX, Audio) or URL ingestion |
| `DELETE` | `/api/documents/{doc_id}` | Deletes document chunks and vector records from index |
| `POST` | `/api/query` | Grounded semantic search, evidence retrieval, and 3-phase roadmap |
| `POST` | `/api/quiz/generate` | Generates document-grounded MCQ questions |
| `POST` | `/api/quiz/evaluate` | Evaluates submitted answers, computes score, and updates SM-2 schedule |
| `GET` | `/api/graph` | Extracts entity nodes and semantic relationship edges |
| `GET` | `/api/analytics` | Returns SuperMemo SM-2 retention metrics and due review dates |

---

## 🐳 Docker Deployment

### Local Docker Build:
```bash
# Build production image
docker build -t memorypal-backend .

# Run container with dynamic port
docker run -p 8501:8501 --env-file .env memorypal-backend
```

### Docker Compose Production Stack:
```bash
docker-compose -f docker-compose.prod.yml up -d --build
```

---

## 👥 Authors & License

* **Authors:** Abhishek Buddiga & Izhaar Ahmed
* **License:** MIT License. See [LICENSE](LICENSE) for details.

# 🧠 MemoryPalAI – Intelligent Knowledge Workspace
### Your Personal AI Tutor • Knowledge Extractor • Quiz Generator • Learning Companion

🔗 **Live Demo:** https://memorypalai.onrender.com  
📦 **Tech Stack:** Streamlit • LangGraph • Gemini 2.5 Flash • Pinecone • SentenceTransformers • PyMuPDF • Python  

---

## 🚀 Overview

**MemoryPalAI** is an Agentic AI Learning System that autonomously:

- Processes **documents, audio, PDFs, URLs**
- Extracts **text + style + tags + topics**
- Generates **embeddings** and stores them in Pinecone
- Performs **RAG (retrieval augmented generation)** for grounded answering
- Generates **quizzes** based on retrieved content
- Evaluates **user answers**
- Creates **revision notes** for weak topics
- Maintains long-term **user performance memory**

It acts as a complete **AI learning companion**.

---

## 🗂️ Features

### 📥 Smart Ingestion
- Supports PDF, TXT, audio (mp3/wav/m4a), URLs  
- Automatic text extraction  
- Detects **subject, style, tone, tags, topic**  
- Splits text using **RecursiveCharacterTextSplitter**  
- Embeds with **Gemini text-embedding-004**

---

### 🔎 Retrieval-Augmented QA
- Retrieves top-k relevant chunks from Pinecone  
- Answers ONLY using retrieved context  
- If unavailable → *“I don't know based on the provided documents.”*

---

### 🧠 Auto Learning Plan
Generates a **3-phase learning plan** from your query and context.

---

### 🧩 Quiz Agent
- Creates quizzes from retrieved content  
- Fully interactive MCQ interface  

---

### 📊 Quiz Evaluation
- Explains each answer  
- Computes final score  
- Highlights weak concepts  

---

### 🔁 Revision Agent
Triggered when score < 70%:

- Generates **concise revision notes**  
- Suggests **YouTube search links**  
- Provides **self-check questions**  
- Stores revision history per topic  

---

### 🧵 Knowledge Graph (Lite)
- Builds relationships between topics and tags  
- Visualized using custom graph UI  

---

## 🏗️ Architecture

```
                ┌─────────────────────────────────────┐
                │            User Uploads             │
                │ (PDF / Text / Audio / URL Input)    │
                └─────────────────────────────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   Ingestion Agent   │
                     │ Text Extraction +   │
                     │ Chunking + Tagging  │
                     └─────────────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │  Embedding Service  │
                     │ (Gemini Embeddings) │
                     └─────────────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │     Pinecone DB     │
                     │  Vector + Metadata  │
                     └─────────────────────┘
                                │
                    ┌──────────┴──────────┐
                    ▼                     ▼
        ┌───────────────────┐   ┌─────────────────────┐
        │ Retrieve & Answer │   │   Knowledge Graph    │
        └───────────────────┘   └─────────────────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │    Quiz Generator   │
          └─────────────────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │  Quiz Evaluation    │
          └─────────────────────┘
                    │
              Score < 70% ?
                    │
         ┌──────────┴───────────┐
         ▼                      ▼
┌────────────────────┐    ┌────────────────────┐
│  Revision Agent     │    │   End of Cycle     │
│ Weak Topic Notes    │    │   Good Score!      │
└────────────────────┘    └────────────────────┘
```
<img width="842" height="447" alt="Screenshot 2025-11-14 at 11 44 22 AM" src="https://github.com/user-attachments/assets/62d4535e-6091-47bf-8e0d-04a56f06ae67" />

---

## 🧑‍💻 Tech Stack

**Frontend**
- Streamlit  
- Custom UI components  

**Backend**
- Python  
- LangGraph Agents  
- Gemini 2.5 Flash  
- Sentence Transformers  
- PyMuPDF  

**Database**
- Pinecone Vector DB  
- Local JSON memory  

---

## ⚙️ Installation

### 1. Clone Repo
```bash
git clone https://github.com/your-username/MemoryPalAI.git
cd MemoryPalAI
```

### 2. Create Environment
```bash
conda create -n memorypal python=3.11 -y
conda activate memorypal
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add `.env`
```
GOOGLE_API_KEY=your_key
GOOGLE_API_KEY_2=optional_secondary_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=memorypal-ai
PINECONE_NAMESPACE=memorypal
```

---

## ▶️ Running the App

### Local:
```bash
streamlit run frontend/streamlit_app.py
```

### Render Deployment:
```bash
streamlit run frontend/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## 🌐 Deployment

Deployed at:  
👉 **https://memorypalai.onrender.com**

---

## Agentic AI Evaluation

### Agentic Characteristics
| Characteristic | Implementation |
|----------------|----------------|
| Autonomous | Full ingestion → retrieval → quiz → revision loop |
| Goal Oriented | Learning goal–driven reasoning |
| Planning | Auto 3-phase learning plan |
| Reasoning | Retrieval-grounded responses |
| Adaptability | Adaptive revision notes |
| Content Awareness | Style, tone, topic, tag extraction |

---

### System Components
| Component | Description |
|----------|-------------|
| Brain | Gemini Flash 2.5 |
| Orchestrator | LangGraph |
| Tools | Embedding, Style Detector, Pinecone, Graph |
| Memory | Pinecone + user_profile.json |
| HITL | User uploads, quiz answering |

---

## 👥 Team Contributions

- **Abhishek (AD23B1012)** — Architected retrieval pipeline, UI, embedding flow.  
- **Izhaar Ahmed (23BDS053)** — Implemented ingestion, style/tone/tag extraction, Pinecone indexing.  
- **Jashwanth (AD23B1020)** — Built Quiz Agent & Evaluation Agent workflows.  
- **Swarup G L (AD23B1020)** — Developed knowledge graph, session manager, and agent orchestration.

---

## Conclusion

MemoryPalAI is a complete autonomous learning system powered by Agentic AI.  
It processes knowledge, quizzes users, evaluates performance, and generates personalized revision notes.

---

## Contribute to MemoryPalAI

MemoryPalAI is fully **open to contributions**!  
We welcome improvements, bug fixes, new ideas, and feature enhancements.

If you find this project useful, please **⭐ star the repository** —  
your support motivates us and helps the project grow! 🚀

Feel free to submit issues or pull requests anytime.  
We’re excited to see your contributions! ❤️

---


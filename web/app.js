// web/app.js
// MemoryPal AI — Next-Gen Cognitive Learning OS
// High-Performance Native React 18 Application (Zero Babel Transpilation Lag)

import React, { useState, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import htm from "htm";

const html = htm.bind(React.createElement);
const API_BASE = "";

// -----------------------------------------------------------------------------
// Pure Vector Icons (Zero network dependencies, instant rendering)
// -----------------------------------------------------------------------------
const Icon = {
  Brain: () => html`
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>
      <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>
      <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>
      <path d="M12 18v4"/>
    </svg>
  `,
  Compass: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>
    </svg>
  `,
  Sparkles: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
    </svg>
  `,
  UploadCloud: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/>
      <path d="M12 12v9"/>
      <path d="m16 16-4-4-4 4"/>
    </svg>
  `,
  Search: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8"/>
      <path d="m21 21-4.3-4.3"/>
    </svg>
  `,
  BookOpen: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
    </svg>
  `,
  Award: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="8" r="6"/>
      <path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/>
    </svg>
  `,
  Share2: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="18" cy="5" r="3"/>
      <circle cx="6" cy="12" r="3"/>
      <circle cx="18" cy="19" r="3"/>
      <line x1="8.59" x2="15.42" y1="13.51" y2="17.49"/>
      <line x1="15.41" x2="8.59" y1="6.51" y2="10.49"/>
    </svg>
  `,
  BarChart3: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3v18h18"/>
      <path d="M18 17V9"/>
      <path d="M13 17V5"/>
      <path d="M8 17v-3"/>
    </svg>
  `,
  CheckCircle2: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <path d="m9 12 2 2 4-4"/>
    </svg>
  `,
  AlertCircle: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" x2="12" y1="8" y2="12"/>
      <line x1="12" x2="12.01" y1="16" y2="16"/>
    </svg>
  `,
  Zap: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
    </svg>
  `,
  Trash2: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18"/>
      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>
      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>
      <line x1="10" x2="10" y1="11" y2="17"/>
      <line x1="14" x2="14" y1="11" y2="17"/>
    </svg>
  `,
  RefreshCw: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
      <path d="M21 3v5h-5"/>
      <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
      <path d="M8 16H3v5"/>
    </svg>
  `,
  Globe: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>
      <path d="M2 12h20"/>
    </svg>
  `,
  Clock: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <polyline points="12 6 12 12 16 14"/>
    </svg>
  `,
  Play: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="5 3 19 12 5 21 5 3"/>
    </svg>
  `,
  ArrowRight: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14"/>
      <path d="m12 5 7 7-7 7"/>
    </svg>
  `,
  Check: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  `,
  X: () => html`
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6 6 18"/>
      <path d="m6 6 12 12"/>
    </svg>
  `
};

// -----------------------------------------------------------------------------
// Resilient Error Boundary (Prevents blank screens on any UI exception)
// -----------------------------------------------------------------------------
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, errorInfo) {
    console.warn("ErrorBoundary caught error:", error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return html`
        <div className="min-h-screen bg-[#030712] flex items-center justify-center p-6 text-slate-100">
          <div className="glass-card max-w-lg w-full p-8 rounded-3xl border border-rose-500/30 bg-rose-950/20 space-y-5 text-center shadow-2xl">
            <div className="w-14 h-14 mx-auto rounded-2xl bg-rose-500/20 text-rose-400 flex items-center justify-center text-2xl font-bold">
              ⚠️
            </div>
            <h2 className="text-xl font-bold text-white">Application Notice</h2>
            <p className="text-xs text-slate-300 leading-relaxed">
              ${this.state.error?.message || "An unexpected UI state occurred."}
            </p>
            <button
              onClick=${() => {
                this.setState({ hasError: false, error: null });
                window.location.reload();
              }}
              className="px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-semibold text-white text-xs shadow-md transition-all"
            >
              Reload & Continue
            </button>
          </div>
        </div>
      `;
    }
    return this.props.children;
  }
}

// -----------------------------------------------------------------------------
// Lightweight Markdown & Interactive Link Renderer
// -----------------------------------------------------------------------------
function renderLineLinks(text) {
  if (!text) return "";
  const regex = /\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\)/g;
  const parts = [];
  let lastIdx = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push(text.substring(lastIdx, match.index));
    }
    const label = match[1];
    const url = match[2];
    const isYoutube = url.includes("youtube.com") || url.includes("youtu.be");

    parts.push(html`
      <a
        key=${match.index}
        href=${url}
        target="_blank"
        rel="noopener noreferrer"
        className=${isYoutube
          ? "inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-500/15 text-rose-300 hover:bg-rose-500/25 border border-rose-500/30 transition-all mx-1"
          : "text-indigo-400 hover:text-indigo-300 underline font-medium mx-0.5"}
      >
        ${isYoutube && html`<span>▶</span>`}
        <span>${label}</span>
      </a>
    `);
    lastIdx = regex.lastIndex;
  }

  if (lastIdx < text.length) {
    parts.push(text.substring(lastIdx));
  }

  return parts.length > 0 ? parts : text;
}

function MarkdownContent({ content }) {
  if (!content) return null;
  const str = String(content);
  const lines = str.split("\n");

  return html`
    <div className="space-y-2 text-sm leading-relaxed text-slate-200">
      ${lines.map((line, idx) => {
        const trimmed = line.trim();
        if (!trimmed) return html`<div key=${idx} className="h-1.5"></div>`;

        if (trimmed.startsWith("### ")) {
          return html`
            <h4 key=${idx} className="text-base font-bold text-indigo-300 pt-2 pb-1 border-b border-white/5 flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
              <span>${trimmed.replace(/^###\s+/, "")}</span>
            </h4>
          `;
        }

        if (trimmed.startsWith("## ")) {
          return html`
            <h3 key=${idx} className="text-lg font-bold text-white pt-3 pb-1 border-b border-white/10">
              ${trimmed.replace(/^##\s+/, "")}
            </h3>
          `;
        }

        if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
          const itemText = trimmed.replace(/^[-*]\s+/, "");
          return html`
            <div key=${idx} className="flex items-start gap-2 pl-2">
              <span className="text-indigo-400 mt-1">•</span>
              <span className="flex-1">${renderLineLinks(itemText)}</span>
            </div>
          `;
        }

        return html`<p key=${idx}>${renderLineLinks(trimmed)}</p>`;
      })}
    </div>
  `;
}

// -----------------------------------------------------------------------------
// Main Application Component
// -----------------------------------------------------------------------------
function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [health, setHealth] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [topics, setTopics] = useState([]);
  const [analytics, setAnalytics] = useState(null);

  // Ingest State
  const [uploadFile, setUploadFile] = useState(null);
  const [urlInput, setUrlInput] = useState("");
  const [topicInput, setTopicInput] = useState("");
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestSuccess, setIngestSuccess] = useState(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState(null);

  // Chat / Query State
  const [queryText, setQueryText] = useState("");
  const [selectedTopic, setSelectedTopic] = useState("All Topics");
  const [isQuerying, setIsQuerying] = useState(false);
  const [queryResult, setQueryResult] = useState(null);
  const [queryError, setQueryError] = useState(null);
  const [showCitationDrawer, setShowCitationDrawer] = useState(false);

  // Quiz Arena State
  const [quizTopic, setQuizTopic] = useState("");
  const [numQuestions, setNumQuestions] = useState(4);
  const [isGeneratingQuiz, setIsGeneratingQuiz] = useState(false);
  const [quizData, setQuizData] = useState(null);
  const [userSelectedAnswers, setUserSelectedAnswers] = useState({});
  const [isEvaluatingQuiz, setIsEvaluatingQuiz] = useState(false);
  const [evaluationResult, setEvaluationResult] = useState(null);

  // Graph State
  const [graphData, setGraphData] = useState(null);
  const [isLoadingGraph, setIsLoadingGraph] = useState(false);

  // Fetch initial telemetry & document inventories
  useEffect(() => {
    fetchHealth();
    fetchDocuments();
    fetchAnalytics();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
        if (data.topics && data.topics.length > 0 && !quizTopic) {
          setQuizTopic(data.topics[0]);
        }
      }
    } catch (e) {
      console.warn("Telemetry offline:", e);
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
        setTopics(data.topics || []);
      }
    } catch (e) {
      console.warn("Documents fetch error:", e);
    }
  };

  const fetchAnalytics = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/analytics`);
      if (res.ok) {
        const data = await res.json();
        setAnalytics(data);
      }
    } catch (e) {
      console.warn("Analytics fetch error:", e);
    }
  };

  const fetchGraph = async () => {
    setIsLoadingGraph(true);
    try {
      const res = await fetch(`${API_BASE}/api/graph`);
      if (res.ok) {
        const data = await res.json();
        setGraphData(data);
      }
    } catch (e) {
      console.warn("Graph fetch error:", e);
    } finally {
      setIsLoadingGraph(false);
    }
  };

  // Ingestion Action
  const handleIngest = async (e) => {
    e.preventDefault();
    if (!uploadFile && !urlInput.trim()) {
      alert("Please upload a file or enter a valid URL.");
      return;
    }

    setIsIngesting(true);
    setIngestSuccess(null);

    const formData = new FormData();
    if (uploadFile) {
      formData.append("file", uploadFile);
      if (uploadFile.name.match(/\.(mp3|wav|m4a)$/i)) {
        setAudioPreviewUrl(URL.createObjectURL(uploadFile));
      } else {
        setAudioPreviewUrl(null);
      }
    }
    if (urlInput.trim()) formData.append("url", urlInput.trim());
    if (topicInput.trim()) formData.append("topic", topicInput.trim());

    try {
      const res = await fetch(`${API_BASE}/api/ingest`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setIngestSuccess(data);
        fetchDocuments();
        fetchHealth();
        setUploadFile(null);
        setUrlInput("");
      } else {
        alert(data.detail || "Ingestion error");
      }
    } catch (err) {
      alert("Ingestion failed: " + err.message);
    } finally {
      setIsIngesting(false);
    }
  };

  // Document Deletion
  const handleDeleteDocument = async (docId) => {
    if (!confirm(`Delete "${docId}" from your knowledge base?`)) return;
    try {
      const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(docId)}`, {
        method: "DELETE"
      });
      if (res.ok) {
        fetchDocuments();
        fetchHealth();
      }
    } catch (e) {
      alert("Delete failed: " + e.message);
    }
  };

  // Query / Grounded QA Action
  const handleQuery = async (e) => {
    if (e && e.preventDefault) e.preventDefault();
    if (!queryText.trim()) return;

    setActiveTab("chat");
    setIsQuerying(true);
    setQueryResult(null);
    setQueryError(null);

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: queryText.trim(),
          topic: selectedTopic === "All Topics" ? null : selectedTopic,
          top_k: 4
        })
      });
      const data = await res.json();
      if (res.ok) {
        setQueryResult(data);
        if (data.subject && !quizTopic) setQuizTopic(data.subject);
      } else {
        setQueryError(data.detail || "Cognitive synthesis failed. Please try again.");
      }
    } catch (err) {
      setQueryError("Query failed: " + err.message);
    } finally {
      setIsQuerying(false);
    }
  };

  // Quiz Generation Action
  const handleGenerateQuiz = async () => {
    const targetTopic = quizTopic.trim() || "General Comprehension";
    setIsGeneratingQuiz(true);
    setQuizData(null);
    setEvaluationResult(null);
    setUserSelectedAnswers({});

    try {
      const res = await fetch(`${API_BASE}/api/quiz/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subject: targetTopic,
          context: queryResult ? queryResult.context : "",
          num_questions: numQuestions
        })
      });
      const data = await res.json();
      if (res.ok) {
        setQuizData(data);
      } else {
        alert(data.detail || "Quiz generation failed");
      }
    } catch (err) {
      alert("Quiz generation failed: " + err.message);
    } finally {
      setIsGeneratingQuiz(false);
    }
  };

  // Quiz Evaluation Action
  const handleSubmitQuiz = async () => {
    if (!quizData) return;

    const letterAnswers = {};
    quizData.questions.forEach((q) => {
      const chosen = userSelectedAnswers[q.id];
      if (chosen) {
        const idx = q.options.indexOf(chosen);
        letterAnswers[q.id] = ["A", "B", "C", "D"][idx] || "A";
      } else {
        letterAnswers[q.id] = "A";
      }
    });

    setIsEvaluatingQuiz(true);
    try {
      const res = await fetch(`${API_BASE}/api/quiz/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subject: quizData.subject,
          user_answers: letterAnswers,
          raw_quiz: quizData.raw_quiz
        })
      });
      const data = await res.json();
      if (res.ok) {
        setEvaluationResult(data);
        fetchAnalytics();
      } else {
        alert(data.detail || "Quiz evaluation failed");
      }
    } catch (err) {
      alert("Evaluation failed: " + err.message);
    } finally {
      setIsEvaluatingQuiz(false);
    }
  };

  return html`
    <div className="min-h-screen flex flex-col bg-[#030712] text-slate-100 bg-grid-pattern relative">
      <div className="absolute top-0 left-0 right-0 h-[500px] spotlight-radial pointer-events-none"></div>

      <!-- ---------------- Floating Island Header ---------------- -->
      <header className="sticky top-0 z-50 glass-nav px-4 sm:px-8 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3 cursor-pointer" onClick=${() => setActiveTab("overview")}>
          <img src="/assets/icon.png" alt="MemoryPal Agent" className="w-10 h-10 object-contain drop-shadow-[0_0_15px_rgba(34,180,173,0.5)]" />
          <div>
            <div className="font-extrabold text-lg tracking-tight flex items-center gap-2">
              <span className="bg-gradient-to-r from-white via-slate-100 to-indigo-200 bg-clip-text text-transparent font-black">MemoryPal</span>
              <span className="text-[11px] font-black tracking-widest text-teal-400 uppercase font-mono px-1.5 py-0.5 rounded bg-teal-500/10 border border-teal-500/20">AGENT</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 hidden sm:inline">
                PRO
              </span>
            </div>
            <div className="text-[11px] text-slate-400 font-medium hidden sm:block">
              Cognitive Learning OS
            </div>
          </div>
        </div>

        <!-- Navigation Tabs (Desktop) -->
        <nav className="hidden lg:flex items-center gap-1.5 bg-slate-900/80 p-1.5 rounded-xl border border-white/5 text-xs font-semibold">
          ${[
            { id: "overview", label: "Overview", icon: Icon.Compass },
            { id: "ingest", label: "Ingest Studio", icon: Icon.UploadCloud },
            { id: "chat", label: "Cognitive Tutor", icon: Icon.BookOpen },
            { id: "quiz", label: "Quiz Arena", icon: Icon.Award },
            { id: "graph", label: "Concept Graph", icon: Icon.Share2 },
            { id: "analytics", label: "Active Recall", icon: Icon.BarChart3 },
          ].map((item) => {
            const active = activeTab === item.id;
            return html`
              <button
                key=${item.id}
                onClick=${() => {
                  setActiveTab(item.id);
                  if (item.id === "graph" && !graphData) fetchGraph();
                }}
                className=${`flex items-center gap-2 px-3.5 py-1.5 rounded-lg transition-all ${
                  active
                    ? "bg-indigo-600 text-white shadow-[0_0_15px_rgba(99,102,241,0.45)]"
                    : "text-slate-400 hover:text-slate-200 hover:bg-white/5"
                }`}
              >
                ${item.label}
              </button>
            `;
          })}
        </nav>

        <!-- Status Beacon -->
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shadow-sm">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>Neural Core Active</span>
          </div>

          <button
            onClick=${() => setActiveTab("ingest")}
            className="hidden sm:flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md transition-all shimmer-btn"
          >
            <${Icon.UploadCloud} />
            <span>Add Source</span>
          </button>
        </div>
      </header>

      <!-- Mobile Navigation Row -->
      <div className="lg:hidden flex overflow-x-auto gap-2 p-2.5 bg-slate-950/90 border-b border-white/5 text-xs">
        ${["overview", "ingest", "chat", "quiz", "graph", "analytics"].map((t) => html`
          <button
            key=${t}
            onClick=${() => setActiveTab(t)}
            className=${`px-3 py-1.5 rounded-lg whitespace-nowrap capitalize font-medium ${
              activeTab === t ? "bg-indigo-600 text-white font-semibold shadow-sm" : "text-slate-400 bg-white/5"
            }`}
          >
            ${t === "chat" ? "Cognitive Tutor" : t === "ingest" ? "Ingest Studio" : t}
          </button>
        `)}
      </div>

      <!-- ---------------- Main Workspace Viewports ---------------- -->
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-10">

        <!-- ==================== VIEW 1: OVERVIEW & BENTO SHOWCASE ==================== -->
        ${activeTab === "overview" && html`
          <div className="space-y-12">
            <!-- Hero Spotlight Headline -->
            <div className="relative rounded-3xl p-8 sm:p-14 overflow-hidden border border-white/10 bg-gradient-to-b from-slate-900/90 via-slate-900/70 to-slate-950/90 shadow-2xl flex flex-col lg:flex-row items-center justify-between gap-8">
              <div className="relative z-10 max-w-2xl space-y-5">
                <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold bg-teal-500/10 text-teal-300 border border-teal-500/30">
                  <${Icon.Sparkles} />
                  <span>Autonomous Cognitive Learning OS</span>
                </div>

                <h1 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight leading-[1.12] bg-gradient-to-r from-white via-slate-100 to-teal-300 bg-clip-text text-transparent">
                  Master Complex Subjects With Neural Precision
                </h1>

                <p className="text-slate-400 text-sm sm:text-base leading-relaxed max-w-xl">
                  Transform academic textbooks, recorded lectures, technical research, and web articles 
                  into an interconnected cognitive learning system with grounded citation RAG, conceptual 
                  quizzes, auto-synthesized revision decks, and <strong>SuperMemo SM-2 Spaced Repetition</strong>.
                </p>

                <!-- Interactive Quick Query Omnibar in Hero -->
                <form onSubmit=${handleQuery} className="pt-2 flex flex-col sm:flex-row gap-2 max-w-xl">
                  <div className="relative flex-1">
                    <div className="absolute left-3.5 top-3.5 text-slate-500">
                      <${Icon.Search} />
                    </div>
                    <input
                      type="text"
                      value=${queryText}
                      onChange=${(e) => setQueryText(e.target.value)}
                      placeholder="Ask any concept from your notes..."
                      className="w-full pl-10 pr-4 py-3 rounded-xl bg-slate-950/90 border border-slate-700 text-sm focus:outline-none focus:border-indigo-500 text-slate-100 placeholder:text-slate-500 shadow-inner"
                    />
                  </div>
                  <button
                    type="submit"
                    onClick=${() => setActiveTab("chat")}
                    className="px-5 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-semibold text-white text-sm shadow-[0_0_20px_rgba(99,102,241,0.4)] transition-all flex items-center justify-center gap-2 whitespace-nowrap"
                  >
                    <span>Synthesize</span>
                    <${Icon.ArrowRight} />
                  </button>
                </form>
              </div>

              <!-- Brand Emblem Showcase -->
              <div className="hidden lg:flex items-center justify-center relative shrink-0">
                <div className="w-48 h-48 rounded-full bg-teal-500/15 blur-3xl absolute inset-0 pointer-events-none"></div>
                <img src="/assets/icon.png" alt="MemoryPal Agent Emblem" className="w-44 h-44 object-contain drop-shadow-[0_0_35px_rgba(34,180,173,0.4)] animate-float relative z-10" />
              </div>
            </div>

            <!-- 4 Quick Telemetry Cards -->
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
              <div className="glass-card glass-card-hover rounded-2xl p-5 border border-white/10">
                <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Indexed Knowledge Vectors
                </div>
                <div className="text-2xl sm:text-3xl font-extrabold text-white mt-1.5">
                  ${health?.total_chunks || 0}
                </div>
                <div className="text-xs text-indigo-400 mt-1 flex items-center gap-1 font-medium">
                  <${Icon.Zap} /> Zero-Latency Vector Engine
                </div>
              </div>

              <div className="glass-card glass-card-hover rounded-2xl p-5 border border-white/10">
                <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Active Knowledge Trees
                </div>
                <div className="text-2xl sm:text-3xl font-extrabold text-white mt-1.5">
                  ${health?.topics?.length || 0}
                </div>
                <div className="text-xs text-violet-400 mt-1 flex items-center gap-1 font-medium">
                  <${Icon.Brain} /> Multi-Cluster Taxonomies
                </div>
              </div>

              <div className="glass-card glass-card-hover rounded-2xl p-5 border border-white/10">
                <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Quizzes Evaluated
                </div>
                <div className="text-2xl sm:text-3xl font-extrabold text-white mt-1.5">
                  ${analytics?.total_quizzes || 0}
                </div>
                <div className="text-xs text-cyan-400 mt-1 flex items-center gap-1 font-medium">
                  <${Icon.Award} /> Active Recall Arena
                </div>
              </div>

              <div className="glass-card glass-card-hover rounded-2xl p-5 border border-white/10">
                <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Memory Retention
                </div>
                <div className="text-2xl sm:text-3xl font-extrabold text-white mt-1.5">
                  ${analytics?.overall_retention || 100}%
                </div>
                <div className="text-xs text-emerald-400 mt-1 flex items-center gap-1 font-medium">
                  <${Icon.CheckCircle2} /> SuperMemo SM-2 Active
                </div>
              </div>
            </div>

            <!-- Bento Grid Architecture Showcase (Reference: rescomail / vinayweb) -->
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
                    Cognitive System Architecture
                  </h2>
                  <p className="text-xs sm:text-sm text-slate-400 mt-1">
                    Engineered for high-retention learning without cognitive overload.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                <!-- Bento 1: Multimodal Ingestion -->
                <div className="glass-card glass-card-hover rounded-2xl p-6 sm:p-7 flex flex-col justify-between">
                  <div className="space-y-3">
                    <div className="w-10 h-10 rounded-xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center border border-indigo-500/20">
                      <${Icon.UploadCloud} />
                    </div>
                    <h3 className="font-bold text-base text-white">Multimodal Neural Ingestion</h3>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Instant text and audio extraction from PDF, DOCX, TXT, and MP3/WAV/M4A recordings. 
                      Classifies pedagogy into Explanatory, Technical, and Example-driven styles.
                    </p>
                  </div>
                  <div className="mt-5 pt-3 border-t border-white/5 flex gap-2 flex-wrap">
                    <span className="px-2.5 py-1 rounded-md text-[11px] font-mono bg-slate-900 text-slate-300 border border-white/5">PyMuPDF</span>
                    <span className="px-2.5 py-1 rounded-md text-[11px] font-mono bg-slate-900 text-slate-300 border border-white/5">Whisper Audio</span>
                    <span className="px-2.5 py-1 rounded-md text-[11px] font-mono bg-slate-900 text-slate-300 border border-white/5">URL Scraper</span>
                  </div>
                </div>

                <!-- Bento 2: Grounded RAG & Roadmaps -->
                <div className="glass-card glass-card-hover rounded-2xl p-6 sm:p-7 flex flex-col justify-between">
                  <div className="space-y-3">
                    <div className="w-10 h-10 rounded-xl bg-cyan-500/10 text-cyan-400 flex items-center justify-center border border-cyan-500/20">
                      <${Icon.BookOpen} />
                    </div>
                    <h3 className="font-bold text-base text-white">Grounded Citations & Roadmaps</h3>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Answers are grounded strictly on source chunks with transparent confidence scores. 
                      Synthesizes a 3-phase mastery study roadmap with direct YouTube tutorial search links.
                    </p>
                  </div>
                  <div className="mt-5 pt-3 border-t border-white/5 flex gap-2 flex-wrap">
                    <span className="px-2.5 py-1 rounded-md text-[11px] font-mono bg-slate-900 text-slate-300 border border-white/5">Strict Grounding</span>
                    <span className="px-2.5 py-1 rounded-md text-[11px] font-mono bg-slate-900 text-slate-300 border border-white/5">3-Phase Mastery</span>
                    <span className="px-2.5 py-1 rounded-md text-[11px] font-mono bg-slate-900 text-slate-300 border border-white/5">Curated Tutorials</span>
                  </div>
                </div>

                <!-- Bento 3: Adaptive Quiz & Auto-Revision -->
                <div className="glass-card glass-card-hover rounded-2xl p-6 sm:p-7 flex flex-col justify-between">
                  <div className="space-y-3">
                    <div className="w-10 h-10 rounded-xl bg-purple-500/10 text-purple-400 flex items-center justify-center border border-purple-500/20">
                      <${Icon.Award} />
                    </div>
                    <h3 className="font-bold text-base text-white">Adaptive Quiz Arena</h3>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      MCQ active recall evaluations. If comprehension falls below 70%, 
                      MemoryPal AI automatically synthesizes a tailored Revision Deck with self-check flashcards.
                    </p>
                  </div>
                  <div className="mt-5 pt-3 border-t border-white/5 flex gap-2 flex-wrap">
                    <span className="px-2.5 py-1 rounded-md text-[11px] font-mono bg-slate-900 text-slate-300 border border-white/5">Active Recall</span>
                    <span className="px-2.5 py-1 rounded-md text-[11px] font-mono bg-slate-900 text-slate-300 border border-white/5">Auto-Revision Deck</span>
                    <span className="px-2.5 py-1 rounded-md text-[11px] font-mono bg-slate-900 text-slate-300 border border-white/5">Pedagogical Scoring</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        `}

        <!-- ==================== VIEW 2: INGESTION STUDIO ==================== -->
        ${activeTab === "ingest" && html`
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                <${Icon.UploadCloud} />
                <span>Knowledge Ingestion Studio</span>
              </h2>
              <p className="text-sm text-slate-400 mt-1">
                Upload PDFs, recorded lectures, DOCX notes, or public web articles.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              <!-- Upload Form (7 cols) -->
              <div className="lg:col-span-7 space-y-5">
                <form onSubmit=${handleIngest} className="glass-card rounded-2xl p-6 space-y-5 border border-white/10">
                  <!-- File Drag & Drop Dropzone -->
                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-2 uppercase tracking-wider">
                      Study Document / Audio Lecture
                    </label>
                    <div className="border-2 border-dashed border-slate-700 hover:border-indigo-500/50 rounded-xl p-8 text-center cursor-pointer transition-colors bg-slate-900/40 relative">
                      <input
                        type="file"
                        onChange=${(e) => setUploadFile(e.target.files[0] || null)}
                        accept=".txt,.pdf,.docx,.mp3,.wav,.m4a"
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                      />
                      <div className="flex flex-col items-center justify-center gap-2.5">
                        <div className="w-10 h-10 rounded-full bg-indigo-500/10 text-indigo-400 flex items-center justify-center border border-indigo-500/20">
                          <${Icon.UploadCloud} />
                        </div>
                        <div className="text-sm font-semibold text-slate-200">
                          ${uploadFile ? uploadFile.name : "Select or drag file to ingest"}
                        </div>
                        <div className="text-xs text-slate-500">
                          PDF, Word DOCX, TXT, or Audio MP3/WAV/M4A (Max 100MB)
                        </div>
                      </div>
                    </div>
                  </div>

                  <!-- Audio Preview Player -->
                  ${audioPreviewUrl && html`
                    <div className="p-3.5 bg-slate-900 rounded-xl border border-white/10 space-y-1.5">
                      <div className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                        <${Icon.Play} />
                        <span>Lecture Audio Preview:</span>
                      </div>
                      <audio controls src=${audioPreviewUrl} className="w-full h-9 rounded" />
                    </div>
                  `}

                  <!-- URL Input -->
                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-2 uppercase tracking-wider">
                      Or Public Web Research URL
                    </label>
                    <div className="relative">
                      <div className="absolute left-3.5 top-3 text-slate-500">
                        <${Icon.Globe} />
                      </div>
                      <input
                        type="url"
                        value=${urlInput}
                        onChange=${(e) => setUrlInput(e.target.value)}
                        placeholder="https://en.wikipedia.org/wiki/Neural_network"
                        className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-slate-900/80 border border-slate-700 text-sm focus:outline-none focus:border-indigo-500 text-slate-200"
                      />
                    </div>
                  </div>

                  <!-- Topic Label -->
                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-2 uppercase tracking-wider">
                      Subject / Topic Category (Optional)
                    </label>
                    <input
                      type="text"
                      value=${topicInput}
                      onChange=${(e) => setTopicInput(e.target.value)}
                      placeholder="e.g. Distributed Computing, Neuroanatomy"
                      className="w-full px-4 py-2.5 rounded-xl bg-slate-900/80 border border-slate-700 text-sm focus:outline-none focus:border-indigo-500 text-slate-200"
                    />
                  </div>

                  <button
                    type="submit"
                    disabled=${isIngesting}
                    className="w-full py-3.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-bold text-white shadow-[0_0_20px_rgba(99,102,241,0.4)] transition-all flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    ${isIngesting ? html`
                      <div className="animate-spin"><${Icon.RefreshCw} /></div>
                      <span>Extracting Style & Embedding Vectors...</span>
                    ` : html`
                      <${Icon.Zap} />
                      <span>Process & Vectorize Document</span>
                    `}
                  </button>
                </form>
              </div>

              <!-- Live Extraction Result & Document Library (5 cols) -->
              <div className="lg:col-span-5 space-y-5">
                ${ingestSuccess && html`
                  <div className="glass-card rounded-2xl p-5 border border-indigo-500/40 bg-indigo-500/5 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="font-bold text-white text-sm flex items-center gap-1.5">
                        <${Icon.CheckCircle2} />
                        <span>Indexed Successfully</span>
                      </div>
                      <span className="text-xs px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 font-mono">
                        ${ingestSuccess.chunks} Chunks
                      </span>
                    </div>

                    <div className="text-xs space-y-1.5 text-slate-300">
                      <div><strong>Source:</strong> ${ingestSuccess.doc_id}</div>
                      <div><strong>Subject:</strong> ${ingestSuccess.subject}</div>
                      <div className="flex gap-2 pt-1">
                        <span className="px-2 py-0.5 rounded text-[11px] bg-purple-500/20 text-purple-300 border border-purple-500/30">
                          Style: ${ingestSuccess.style}
                        </span>
                        <span className="px-2 py-0.5 rounded text-[11px] bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                          Tone: ${ingestSuccess.tone}
                        </span>
                      </div>
                    </div>
                  </div>
                `}

                <!-- Documents Library -->
                <div className="glass-card rounded-2xl p-5 border border-white/10 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-sm text-white">Indexed Documents (${documents.length})</h3>
                    <button
                      onClick=${fetchDocuments}
                      className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
                    >
                      <${Icon.RefreshCw} />
                      <span>Refresh</span>
                    </button>
                  </div>

                  <div className="space-y-2.5 max-h-96 overflow-y-auto pr-1">
                    ${documents.length === 0 ? html`
                      <div className="text-xs text-slate-500 py-6 text-center">
                        No documents indexed yet. Use the upload box on the left.
                      </div>
                    ` : documents.map((d, i) => html`
                      <div
                        key=${i}
                        className="bg-slate-900/80 p-3 rounded-xl border border-white/5 flex items-center justify-between gap-3"
                      >
                        <div className="min-w-0">
                          <div className="text-xs font-semibold text-slate-200 truncate">
                            ${d.document_id}
                          </div>
                          <div className="text-[11px] text-slate-400 flex items-center gap-2 mt-0.5">
                            <span>${d.topic}</span>
                            <span>•</span>
                            <span>${d.chunks} chunks</span>
                          </div>
                        </div>
                        <button
                          onClick=${() => handleDeleteDocument(d.document_id)}
                          className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                          title="Delete Document"
                        >
                          <${Icon.Trash2} />
                        </button>
                      </div>
                    `)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        `}

        <!-- ==================== VIEW 3: COGNITIVE TUTOR & ROADMAP ==================== -->
        ${activeTab === "chat" && html`
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                <${Icon.BookOpen} />
                <span>Cognitive Tutor & 3-Phase Mastery Roadmap</span>
              </h2>
              <p className="text-sm text-slate-400 mt-1">
                Grounded knowledge synthesis with evidence inspections and structured study milestones.
              </p>
            </div>

            <!-- Query Input Bar -->
            <form onSubmit=${handleQuery} className="glass-card rounded-2xl p-4 border border-white/10 flex flex-col sm:flex-row gap-3">
              <div className="flex-1 relative">
                <div className="absolute left-3.5 top-3.5 text-slate-500">
                  <${Icon.Search} />
                </div>
                <input
                  type="text"
                  value=${queryText}
                  onChange=${(e) => setQueryText(e.target.value)}
                  placeholder="Ask any concept question (e.g. How does backpropagation compute gradients?)..."
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-slate-900/90 border border-slate-700 text-sm focus:outline-none focus:border-indigo-500 text-slate-200"
                />
              </div>

              <div className="sm:w-56">
                <select
                  value=${selectedTopic}
                  onChange=${(e) => setSelectedTopic(e.target.value)}
                  className="w-full px-3 py-2.5 rounded-xl bg-slate-900/90 border border-slate-700 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  <option value="All Topics">All Topics</option>
                  ${topics.map((t, idx) => html`
                    <option key=${idx} value=${t}>${t}</option>
                  `)}
                </select>
              </div>

              <button
                type="submit"
                disabled=${isQuerying}
                className="px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-semibold text-white shadow-md transition-all flex items-center justify-center gap-2 whitespace-nowrap disabled:opacity-50"
              >
                ${isQuerying ? html`
                  <div className="animate-spin"><${Icon.RefreshCw} /></div>
                  <span>Synthesizing...</span>
                ` : html`
                  <${Icon.Sparkles} />
                  <span>Synthesize</span>
                `}
              </button>
            </form>

            <!-- Loading Synthesis State -->
            ${isQuerying && html`
              <div className="glass-card rounded-2xl p-8 border border-indigo-500/30 bg-indigo-950/20 space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center animate-spin">
                    <${Icon.RefreshCw} />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-white">Synthesizing Cognitive Knowledge</h4>
                    <p className="text-xs text-indigo-300 mt-0.5">Scanning vector embeddings • Generating evidence citations • Crafting 3-phase roadmap...</p>
                  </div>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <div className="bg-gradient-to-r from-indigo-500 via-cyan-400 to-indigo-500 h-full w-3/4 animate-pulse"></div>
                </div>
              </div>
            `}

            <!-- Query Error State -->
            ${queryError && html`
              <div className="glass-card rounded-2xl p-5 border border-rose-500/30 bg-rose-950/20 flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="text-rose-400 flex-shrink-0"><${Icon.AlertCircle} /></div>
                  <div className="text-xs text-rose-200">${queryError}</div>
                </div>
                <button
                  onClick=${handleQuery}
                  className="px-4 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 text-xs font-semibold whitespace-nowrap"
                >
                  Retry
                </button>
              </div>
            `}

            <!-- Grounded Response Card -->
            ${queryResult && html`
              <div className="space-y-6">
                <div className="glass-card rounded-2xl p-6 sm:p-8 border border-white/10 space-y-4">
                  <div className="flex items-center justify-between pb-3 border-b border-white/5">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-8 rounded-lg bg-indigo-500/20 text-indigo-300 flex items-center justify-center">
                        <${Icon.Brain} />
                      </div>
                      <span className="font-bold text-white text-sm">Grounded Cognitive Synthesis</span>
                    </div>

                    ${queryResult.grounded ? html`
                      <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1">
                        <${Icon.CheckCircle2} /> Grounded in Sources
                      </span>
                    ` : html`
                      <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20 flex items-center gap-1">
                        <${Icon.AlertCircle} /> Foundational Knowledge
                      </span>
                    `}
                  </div>

                  <${MarkdownContent} content=${queryResult.answer} />

                  <!-- Toggle Retrieved Evidence Chunks -->
                  ${queryResult.sources && queryResult.sources.length > 0 && html`
                    <div className="pt-3 border-t border-white/5">
                      <button
                        onClick=${() => setShowCitationDrawer(!showCitationDrawer)}
                        className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
                      >
                        <span>${showCitationDrawer ? "Hide Citation Evidence" : "Inspect Retrieved Citation Evidence & Scores"}</span>
                      </button>

                      ${showCitationDrawer && html`
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-3">
                          ${queryResult.sources.map((src, i) => html`
                            <div key=${i} className="p-3.5 bg-slate-900 rounded-xl border border-white/5 space-y-1.5 text-xs">
                              <div className="flex items-center justify-between font-semibold text-slate-300">
                                <span className="truncate">${src.source}</span>
                                <span className="text-emerald-400">${src.confidence}% Match</span>
                              </div>
                              <div className="text-slate-400 text-[11px] leading-relaxed line-clamp-3">
                                ${src.text}
                              </div>
                            </div>
                          `)}
                        </div>
                      `}
                    </div>
                  `}
                </div>

                <!-- 3-Phase Study Roadmap Card -->
                ${queryResult.roadmap && html`
                  <div className="glass-card rounded-2xl p-6 sm:p-8 border border-white/10 space-y-4">
                    <div className="flex items-center gap-2 pb-3 border-b border-white/5">
                      <div className="w-8 h-8 rounded-lg bg-indigo-500/20 text-indigo-300 flex items-center justify-center">
                        <${Icon.Compass} />
                      </div>
                      <h3 className="font-bold text-white text-base">
                        3-Phase Mastery Study Roadmap & Curated Tutorials
                      </h3>
                    </div>

                    <${MarkdownContent} content=${queryResult.roadmap} />

                    <div className="pt-4 border-t border-white/5 flex justify-end">
                      <button
                        onClick=${() => {
                          setQuizTopic(queryResult.subject || queryText);
                          setActiveTab("quiz");
                        }}
                        className="px-4 py-2 rounded-xl bg-indigo-600/30 hover:bg-indigo-600 text-indigo-200 hover:text-white text-xs font-semibold transition-all flex items-center gap-1.5"
                      >
                        <${Icon.Award} />
                        <span>Launch Quiz on This Topic</span>
                      </button>
                    </div>
                  </div>
                `}
              </div>
            `}
          </div>
        `}

        <!-- ==================== VIEW 4: QUIZ ARENA ==================== -->
        ${activeTab === "quiz" && html`
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                <${Icon.Award} />
                <span>Adaptive Quiz Arena & Auto-Revision</span>
              </h2>
              <p className="text-sm text-slate-400 mt-1">
                Active recall testing with real-time pedagogical breakdowns and automated revision flashcards.
              </p>
            </div>

            <!-- Quiz Setup Box -->
            <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-12 gap-4">
                <div className="sm:col-span-8">
                  <label className="block text-xs font-semibold text-slate-300 mb-1.5 uppercase tracking-wider">
                    Topic of Assessment
                  </label>
                  <input
                    type="text"
                    value=${quizTopic}
                    onChange=${(e) => setQuizTopic(e.target.value)}
                    placeholder="e.g. Backpropagation, Supervised Learning"
                    className="w-full px-4 py-2.5 rounded-xl bg-slate-900/80 border border-slate-700 text-sm focus:outline-none focus:border-indigo-500 text-slate-200"
                  />
                </div>

                <div className="sm:col-span-4">
                  <label className="block text-xs font-semibold text-slate-300 mb-1.5 uppercase tracking-wider">
                    Questions Count
                  </label>
                  <select
                    value=${numQuestions}
                    onChange=${(e) => setNumQuestions(Number(e.target.value))}
                    className="w-full px-3 py-2.5 rounded-xl bg-slate-900/80 border border-slate-700 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                  >
                    <option value=${3}>3 Questions (Quick Assessment)</option>
                    <option value=${4}>4 Questions (Standard Review)</option>
                    <option value=${5}>5 Questions (Comprehensive)</option>
                    <option value=${8}>8 Questions (Mastery Evaluation)</option>
                  </select>
                </div>
              </div>

              <button
                onClick=${handleGenerateQuiz}
                disabled=${isGeneratingQuiz}
                className="w-full py-3.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-bold text-white shadow-[0_0_20px_rgba(99,102,241,0.4)] transition-all flex items-center justify-center gap-2 disabled:opacity-50"
              >
                ${isGeneratingQuiz ? html`
                  <div className="animate-spin"><${Icon.RefreshCw} /></div>
                  <span>Synthesizing Conceptual Questions...</span>
                ` : html`
                  <${Icon.Zap} />
                  <span>Generate Interactive Quiz</span>
                `}
              </button>
            </div>

            <!-- Quiz Questions Stream -->
            ${quizData && quizData.questions && html`
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="font-extrabold text-lg text-white">
                    Topic: <span className="text-indigo-300">${quizData.subject}</span>
                  </h3>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-slate-400 font-medium">
                      ${quizData.questions.length} Questions
                    </span>
                    ${evaluationResult && html`
                      <span className=${`px-2.5 py-1 rounded-full text-xs font-bold ${
                        evaluationResult.score_pct >= 80 ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30" :
                        evaluationResult.score_pct >= 60 ? "bg-amber-500/20 text-amber-300 border border-amber-500/30" :
                        "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                      }`}>
                        Score: ${evaluationResult.score} / ${evaluationResult.total_questions} (${evaluationResult.score_pct}%)
                      </span>
                    `}
                  </div>
                </div>

                <div className="space-y-5">
                  ${quizData.questions.map((q) => {
                    const qBreakdown = evaluationResult?.breakdown?.find(b => b.questionId === q.id);
                    return html`
                      <div key=${q.id} className=${`glass-card rounded-2xl p-6 border transition-all space-y-4 ${
                        qBreakdown
                          ? qBreakdown.isCorrect
                            ? "border-emerald-500/40 bg-emerald-950/10"
                            : "border-rose-500/40 bg-rose-950/10"
                          : "border-white/10"
                      }`}>
                        <div className="font-semibold text-base text-white flex items-start justify-between gap-3">
                          <div className="flex items-start gap-3">
                            <span className="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-mono text-xs mt-0.5">
                              ${q.id}
                            </span>
                            <span>${q.title}</span>
                          </div>

                          ${qBreakdown && html`
                            <span className=${`px-2.5 py-0.5 rounded text-xs font-bold shrink-0 ${
                              qBreakdown.isCorrect
                                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                                : "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                            }`}>
                              ${qBreakdown.isCorrect ? "Correct" : "Incorrect"}
                            </span>
                          `}
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                          ${q.options.map((opt, idx) => {
                            const letter = ["A", "B", "C", "D"][idx];
                            const isSelected = userSelectedAnswers[q.id] === opt;
                            
                            // Post-evaluation highlighting states
                            let btnStyle = "bg-slate-900/60 border-slate-800 text-slate-300 hover:border-slate-700 hover:bg-slate-900";
                            let circleStyle = "bg-slate-800 text-slate-400";
                            let tagBadge = null;

                            if (qBreakdown) {
                              const isCorrectAnswer = (letter === qBreakdown.correctAnswer);
                              const isUserChoice = (letter === qBreakdown.userAnswer);

                              if (isCorrectAnswer) {
                                btnStyle = "bg-emerald-600/20 border-emerald-500 text-emerald-200 shadow-[0_0_15px_rgba(16,185,129,0.25)] font-medium";
                                circleStyle = "bg-emerald-500 text-white font-bold";
                                tagBadge = html`<span className="text-[11px] font-bold text-emerald-400">✓ Correct Answer</span>`;
                              } else if (isUserChoice && !qBreakdown.isCorrect) {
                                btnStyle = "bg-rose-600/20 border-rose-500 text-rose-200 shadow-[0_0_15px_rgba(244,63,94,0.25)] font-medium";
                                circleStyle = "bg-rose-500 text-white font-bold";
                                tagBadge = html`<span className="text-[11px] font-bold text-rose-400">✕ Your Choice</span>`;
                              } else {
                                btnStyle = "bg-slate-900/40 border-slate-800/60 text-slate-500 opacity-60";
                                circleStyle = "bg-slate-800/80 text-slate-600";
                              }
                            } else if (isSelected) {
                              btnStyle = "bg-indigo-600/30 border-indigo-500 text-white shadow-[0_0_15px_rgba(99,102,241,0.3)]";
                              circleStyle = "bg-indigo-500 text-white";
                            }

                            return html`
                              <button
                                key=${idx}
                                type="button"
                                disabled=${!!evaluationResult}
                                onClick=${() => setUserSelectedAnswers({ ...userSelectedAnswers, [q.id]: opt })}
                                className=${`p-3.5 rounded-xl border text-left text-sm transition-all flex items-center justify-between gap-3 ${btnStyle}`}
                              >
                                <div className="flex items-center gap-3">
                                  <div className=${`w-6 h-6 rounded-full flex items-center justify-center font-bold text-xs shrink-0 ${circleStyle}`}>
                                    ${letter}
                                  </div>
                                  <span className="flex-1">${opt}</span>
                                </div>
                                ${tagBadge}
                              </button>
                            `;
                          })}
                        </div>

                        <!-- Grounded Explanation Card -->
                        ${qBreakdown && html`
                          <div className="mt-3 p-4 rounded-xl bg-slate-900/80 border border-indigo-500/20 text-xs text-slate-300 space-y-1">
                            <div className="font-bold text-indigo-400 flex items-center gap-1.5">
                              <span>💡 Grounded Explanation</span>
                            </div>
                            <p className="leading-relaxed text-slate-300">${qBreakdown.explanation}</p>
                          </div>
                        `}
                      </div>
                    `;
                  })}
                </div>

                ${!evaluationResult ? html`
                  <button
                    onClick=${handleSubmitQuiz}
                    disabled=${isEvaluatingQuiz}
                    className="w-full py-4 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400 font-bold text-white shadow-[0_0_25px_rgba(16,185,129,0.35)] transition-all flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    ${isEvaluatingQuiz ? html`
                      <div className="animate-spin"><${Icon.RefreshCw} /></div>
                      <span>Evaluating Answers & Updating Spaced Repetition...</span>
                    ` : html`
                      <${Icon.CheckCircle2} />
                      <span>Submit Answers for Evaluation</span>
                    `}
                  </button>
                ` : html`
                  <button
                    onClick=${() => {
                      setEvaluationResult(null);
                      setUserSelectedAnswers({});
                      handleGenerateQuiz();
                    }}
                    className="w-full py-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-bold text-white shadow-[0_0_20px_rgba(99,102,241,0.4)] transition-all flex items-center justify-center gap-2"
                  >
                    <${Icon.RefreshCw} />
                    <span>Retake or Generate New Assessment</span>
                  </button>
                `}
              </div>
            `}

            <!-- Scorecard & Auto-Revision Deck -->
            ${evaluationResult && html`
              <div className="space-y-6">
                <div className="glass-card rounded-2xl p-6 sm:p-8 border border-white/10 space-y-4">
                  <div className="flex items-center justify-between pb-3 border-b border-white/5">
                    <div>
                      <h3 className="font-bold text-lg text-white">Evaluation Scorecard</h3>
                      <p className="text-xs text-slate-400 mt-0.5">
                        ${evaluationResult.score} of ${evaluationResult.total_questions} questions correct
                      </p>
                    </div>
                    <div className=${`text-3xl font-black ${
                      evaluationResult.score_pct >= 80 ? "text-emerald-400" :
                      evaluationResult.score_pct >= 60 ? "text-amber-400" :
                      "text-rose-400"
                    }`}>
                      ${evaluationResult.score_pct}%
                    </div>
                  </div>

                  <div className="text-slate-200 text-sm leading-relaxed whitespace-pre-line">
                    ${evaluationResult.evaluation}
                  </div>
                </div>

                <!-- Auto-Revision Trigger (< 70%) -->
                ${evaluationResult.revision_pack && html`
                  <div className="glass-card rounded-2xl p-6 sm:p-8 border-l-4 border-rose-500 bg-rose-500/5 space-y-4">
                    <div className="flex items-center gap-2">
                      <div className="text-rose-400"><${Icon.AlertCircle} /></div>
                      <h3 className="font-bold text-rose-300 text-base">
                        Auto-Revision Deck Activated (Score &lt; 70%)
                      </h3>
                    </div>
                    <p className="text-xs text-slate-400">
                      Weak concepts identified. Reinforce with the following synthesized explanations and flashcards:
                    </p>
                    <div className="text-slate-200 text-sm leading-relaxed whitespace-pre-line pt-2">
                      ${evaluationResult.revision_pack}
                    </div>
                  </div>
                `}
              </div>
            `}
          </div>
        `}

        <!-- ==================== VIEW 5: CONCEPT GRAPH ==================== -->
        ${activeTab === "graph" && html`
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                  <${Icon.Share2} />
                  <span>Interactive Concept Knowledge Graph</span>
                </h2>
                <p className="text-sm text-slate-400 mt-1">
                  Entity taxonomy mapping topics, core concepts, and semantic tags.
                </p>
              </div>

              <button
                onClick=${fetchGraph}
                disabled=${isLoadingGraph}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-white/10 flex items-center gap-1.5"
              >
                <${Icon.RefreshCw} />
                <span>Reload Graph</span>
              </button>
            </div>

            <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-4">
              <!-- Entity Legend -->
              <div className="flex flex-wrap gap-4 text-xs font-medium">
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-full bg-purple-500"></span>
                  <span className="text-slate-300">Topic</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-full bg-cyan-400"></span>
                  <span className="text-slate-300">Concept</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-full bg-emerald-400"></span>
                  <span className="text-slate-300">Tag</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-full bg-amber-400"></span>
                  <span className="text-slate-300">Phrase</span>
                </div>
              </div>

              <!-- Interactive Node Clusters -->
              <div className="w-full h-96 bg-slate-950 rounded-xl border border-white/5 relative overflow-hidden flex items-center justify-center p-4">
                ${isLoadingGraph ? html`
                  <div className="flex items-center gap-2 text-sm text-slate-400">
                    <div className="animate-spin text-indigo-400"><${Icon.RefreshCw} /></div>
                    <span>Synthesizing network topology...</span>
                  </div>
                ` : graphData && graphData.nodes && graphData.nodes.length > 0 ? html`
                  <div className="w-full h-full flex flex-wrap gap-3 items-center justify-center p-4 overflow-y-auto">
                    ${graphData.nodes.map((node, idx) => {
                      const colors = {
                        Topic: "bg-purple-500/20 text-purple-300 border-purple-500/30",
                        Concept: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
                        Tag: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
                        Phrase: "bg-amber-500/20 text-amber-300 border-amber-500/30",
                      };
                      const colorClass = colors[node.type] || "bg-indigo-500/20 text-indigo-300 border-indigo-500/30";
                      return html`
                        <div
                          key=${idx}
                          className=${`px-3 py-1.5 rounded-xl border text-xs font-semibold flex items-center gap-1.5 shadow-sm ${colorClass}`}
                        >
                          <span className="w-1.5 h-1.5 rounded-full bg-current"></span>
                          <span>${node.id}</span>
                          <span className="text-[10px] opacity-60">(${node.type})</span>
                        </div>
                      `;
                    })}
                  </div>
                ` : html`
                  <div className="text-xs text-slate-500">
                    No graph entities extracted yet. Ingest documents to populate the concept graph.
                  </div>
                `}
              </div>
            </div>
          </div>
        `}

        <!-- ==================== VIEW 6: RETENTION MEMORY (SM-2) ==================== -->
        ${activeTab === "analytics" && html`
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                <${Icon.BarChart3} />
                <span>SuperMemo SM-2 Spaced Repetition Memory</span>
              </h2>
              <p className="text-sm text-slate-400 mt-1">
                Continuous cognitive decay tracking and automated review scheduling.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <!-- Topic Mastery Breakdown -->
              <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-4">
                <h3 className="font-bold text-base text-white">Subject Mastery Progress</h3>
                <div className="space-y-4">
                  ${analytics?.mastery_list && analytics.mastery_list.length > 0 ? (
                    analytics.mastery_list.map((item, i) => html`
                      <div key=${i} className="space-y-1.5">
                        <div className="flex justify-between text-xs font-semibold text-slate-200">
                          <span>${item.topic}</span>
                          <span>${item.avg_score}% (${item.attempts} attempts)</span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 transition-all duration-500"
                            style=${{ width: `${Math.min(item.avg_score, 100)}%` }}
                          ></div>
                        </div>
                      </div>
                    `)
                  ) : html`
                    <div className="text-xs text-slate-500 py-6 text-center">
                      No quiz history recorded yet. Complete quizzes to build your memory profile!
                    </div>
                  `}
                </div>
              </div>

              <!-- Upcoming Review Schedule -->
              <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-4">
                <h3 className="font-bold text-base text-white">Upcoming Review Schedule</h3>
                <div className="space-y-3">
                  ${analytics?.mastery_list && analytics.mastery_list.length > 0 ? (
                    analytics.mastery_list.map((item, i) => html`
                      <div key=${i} className="bg-slate-900/80 p-3 rounded-xl border border-white/5 flex items-center justify-between text-xs">
                        <div>
                          <div className="font-bold text-slate-200">${item.topic}</div>
                          <div className="text-[11px] text-slate-400 mt-0.5">
                            Status: <span className="text-indigo-300 font-medium">${item.schedule.status || "Progressing"}</span>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-mono text-slate-300">${item.schedule.next_review_date || "Pending"}</div>
                          <div className="text-[10px] text-slate-500">Next Review</div>
                        </div>
                      </div>
                    `)
                  ) : html`
                    <div className="text-xs text-slate-500 py-6 text-center">
                      Schedule will appear as you attempt quizzes.
                    </div>
                  `}
                </div>
              </div>
            </div>
          </div>
        `}

      </main>

      <!-- ---------------- Footer ---------------- -->
      <footer className="border-t border-white/10 py-8 px-4 text-center text-xs text-slate-500 space-y-2.5">
        <div className="flex items-center justify-center gap-2">
          <img src="/assets/icon.png" alt="MemoryPal Agent" className="w-6 h-6 object-contain" />
          <span className="font-extrabold text-slate-200">MemoryPal</span>
          <span className="text-[10px] font-black text-teal-400 uppercase font-mono px-1.5 py-0.5 rounded bg-teal-500/10 border border-teal-500/20">AGENT</span>
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">v2.0</span>
        </div>
        <div>
          Autonomous Cognitive Learning OS • Powered by Gemini 3.8 Flash & Vector Retrieval
        </div>
        <div className="text-[11px] text-slate-600">
          Crafted with glassmorphic design, zero-latency vector retrieval, and SuperMemo SM-2 spaced repetition.
        </div>
      </footer>
    </div>
  `;
}

// Render root with ErrorBoundary protection (prevents blank screen)
const container = document.getElementById("root");
const root = createRoot(container);
root.render(html`
  <${ErrorBoundary}>
    <${App} />
  </${ErrorBoundary}>
`);

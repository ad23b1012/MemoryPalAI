// frontend/lib/api.ts

export interface HealthResponse {
  status: string;
  project: string;
  version: string;
  engine: string;
  engine_ready: boolean;
  vector_backend: string;
  vector_mode: string;
  database: string;
  total_chunks: number;
  total_documents: number;
  topics: string[];
}

export interface DocumentItem {
  document_id: string;
  topic: string;
  chunks: number;
  indexed_at: string;
  style?: string;
  tone?: string;
  tags?: string[];
  snippet?: string;
}

export interface DocumentsResponse {
  documents: DocumentItem[];
  topics: string[];
  total_chunks: number;
  total_documents: number;
  backend: string;
}

export interface CitationSource {
  id: string;
  source: string;
  topic: string;
  confidence: number;
  text: string;
}

export interface QueryResponse {
  answer: string;
  grounded: boolean;
  context: string;
  sources: CitationSource[];
  roadmap: string;
  subject: string;
}

export interface QuizQuestion {
  id: string;
  number: number;
  title: string;
  options: string[];
  answer?: string;
  explanation?: string;
}

export interface QuizGenerateResponse {
  subject: string;
  questions: QuizQuestion[];
  raw_quiz: string;
}

export interface QuestionBreakdown {
  questionId: string;
  userAnswer: string;
  correctAnswer: string;
  isCorrect: boolean;
  explanation: string;
}

export interface QuizEvaluateResponse {
  subject: string;
  score_pct: number;
  score: number;
  total_questions: number;
  evaluation: string;
  breakdown: QuestionBreakdown[];
  revision_pack: string | null;
  schedule: Record<string, any>;
}

export interface GraphNode {
  id: string;
  label: string;
  group?: string;
}

export interface GraphEdge {
  from: string;
  to: string;
  label?: string;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface AnalyticsResponse {
  profile: Record<string, any>;
  due_topics: string[];
  mastery_list: Array<{
    topic: string;
    avg_score: number;
    attempts: number;
    last_score: number;
    revisions_count: number;
    schedule: Record<string, any>;
  }>;
  total_quizzes: number;
  overall_retention: number;
}

const API_BASE = "";

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function fetchDocuments(): Promise<DocumentsResponse> {
  const res = await fetch(`${API_BASE}/api/documents`);
  if (!res.ok) throw new Error("Failed to fetch documents");
  return res.json();
}

export async function deleteDocument(docId: string): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("Failed to delete document");
  return res.json();
}

export async function queryGroundedKnowledge(query: string, topic?: string): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      topic: topic && topic !== "All Topics" ? topic : null,
      top_k: 4,
    }),
  });
  if (!res.ok) throw new Error("Query synthesis failed");
  return res.json();
}

export async function generateQuiz(subject: string, context?: string, numQuestions = 4): Promise<QuizGenerateResponse> {
  const res = await fetch(`${API_BASE}/api/quiz/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      subject,
      context: context || "",
      num_questions: numQuestions,
    }),
  });
  if (!res.ok) throw new Error("Quiz generation failed");
  return res.json();
}

export async function evaluateQuiz(
  subject: string,
  userAnswers: Record<string, string>,
  rawQuiz: string
): Promise<QuizEvaluateResponse> {
  const res = await fetch(`${API_BASE}/api/quiz/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      subject,
      user_answers: userAnswers,
      raw_quiz: rawQuiz,
    }),
  });
  if (!res.ok) throw new Error("Quiz evaluation failed");
  return res.json();
}

export async function fetchGraph(text?: string): Promise<GraphResponse> {
  const url = text ? `${API_BASE}/api/graph?text=${encodeURIComponent(text)}` : `${API_BASE}/api/graph`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch graph");
  return res.json();
}

export async function fetchAnalytics(): Promise<AnalyticsResponse> {
  const res = await fetch(`${API_BASE}/api/analytics`);
  if (!res.ok) throw new Error("Failed to fetch analytics");
  return res.json();
}

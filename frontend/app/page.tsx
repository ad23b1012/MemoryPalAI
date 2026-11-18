"use client";

import React, { useState, useEffect } from "react";
import { Navbar } from "@/components/navbar";
import { Hero } from "@/components/hero";
import { IngestStudio } from "@/components/ingest-studio";
import { CognitiveTutor } from "@/components/cognitive-tutor";
import { QuizArena } from "@/components/quiz-arena";
import { ConceptGraph } from "@/components/concept-graph";
import { ActiveRecall } from "@/components/active-recall";
import { Footer } from "@/components/footer";
import { DocumentItem, fetchHealth, fetchDocuments } from "@/lib/api";

export default function Home() {
  const [activeTab, setActiveTab] = useState("overview");
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [topics, setTopics] = useState<string[]>([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [isEngineReady, setIsEngineReady] = useState(true);
  const [activeQuery, setActiveQuery] = useState("");
  const [activeQuizTopic, setActiveQuizTopic] = useState("Agentic AI");

  const loadData = async () => {
    try {
      const [health, docs] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchDocuments().catch(() => null),
      ]);

      if (health) {
        setIsEngineReady(health.engine_ready);
        if (health.total_chunks) setTotalChunks(health.total_chunks);
      }
      if (docs) {
        setDocuments(docs.documents || []);
        setTopics(docs.topics || []);
        if (docs.total_chunks) setTotalChunks(docs.total_chunks);
      }
    } catch (e) {
      console.warn("Initial data load notice:", e);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleLaunchQuiz = (topic: string) => {
    setActiveQuizTopic(topic);
    setActiveTab("quiz");
  };

  const handleHeroSearch = (query: string) => {
    setActiveQuery(query);
    setActiveTab("chat");
  };

  return (
    <div className="min-h-screen flex flex-col liquid-aurora-bg text-slate-100 liquid-grid-overlay relative overflow-x-hidden selection:bg-teal-400/30 selection:text-teal-200">
      {/* Living Dynamic Floating Liquid Aurora Spheres */}
      <div className="liquid-orb-cyan -top-40 -left-20 animate-liquid-drift-1" />
      <div className="liquid-orb-violet -top-20 -right-20 animate-liquid-drift-2" />
      <div className="liquid-orb-magenta top-1/3 left-1/4 animate-liquid-drift-3" />
      <div className="liquid-orb-emerald top-2/3 -right-20 animate-liquid-pulse-glow" />
      <div className="liquid-orb-cyan -bottom-40 left-10 animate-liquid-drift-2" />
      <div className="liquid-orb-violet -bottom-40 right-20 animate-liquid-drift-1" />

      {/* Floating Island Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isEngineReady={isEngineReady}
      />

      {/* Mobile Tab Scroller Capsule */}
      <div className="lg:hidden px-4 py-2 sticky top-[72px] z-40">
        <div className="flex items-center gap-1.5 overflow-x-auto p-1.5 liquid-glass-dock text-xs">
          {[
            { id: "overview", label: "Overview" },
            { id: "ingest", label: "Ingest Studio" },
            { id: "chat", label: "Cognitive Tutor" },
            { id: "quiz", label: "Quiz Arena" },
            { id: "graph", label: "Concept Graph" },
            { id: "analytics", label: "Active Recall" },
          ].map((t) => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`px-3.5 py-1.5 rounded-full whitespace-nowrap font-bold transition-all select-none ${
                activeTab === t.id
                  ? "liquid-active-pill"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-10 relative z-10">
        {activeTab === "overview" && (
          <Hero
            onSearch={handleHeroSearch}
            setActiveTab={setActiveTab}
            totalDocs={documents.length}
            totalChunks={totalChunks}
          />
        )}

        {activeTab === "ingest" && (
          <IngestStudio
            documents={documents}
            onRefresh={loadData}
          />
        )}

        {activeTab === "chat" && (
          <CognitiveTutor
            initialQuery={activeQuery}
            topics={topics}
            onLaunchQuiz={handleLaunchQuiz}
          />
        )}

        {activeTab === "quiz" && (
          <QuizArena
            initialTopic={activeQuizTopic}
            onQuizEvaluated={loadData}
          />
        )}

        {activeTab === "graph" && (
          <ConceptGraph />
        )}

        {activeTab === "analytics" && (
          <ActiveRecall />
        )}
      </main>

      {/* Global Footer */}
      <Footer />
    </div>
  );
}

"use client";

import React, { useState } from "react";
import { BookOpen, Search, Sparkles, RefreshCw, CheckCircle2, AlertCircle, Compass, Award, ExternalLink, Play } from "lucide-react";
import { QueryResponse, queryGroundedKnowledge } from "@/lib/api";

interface CognitiveTutorProps {
  initialQuery?: string;
  topics: string[];
  onLaunchQuiz: (topic: string) => void;
}

export function CognitiveTutor({ initialQuery = "", topics, onLaunchQuiz }: CognitiveTutorProps) {
  const [queryText, setQueryText] = useState(initialQuery);
  const [selectedTopic, setSelectedTopic] = useState("All Topics");
  const [isQuerying, setIsQuerying] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showCitations, setShowCitations] = useState(false);

  const handleSynthesize = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!queryText.trim()) return;

    setIsQuerying(true);
    setResult(null);
    setErrorMsg(null);

    try {
      const data = await queryGroundedKnowledge(queryText.trim(), selectedTopic);
      setResult(data);
    } catch (err: any) {
      setErrorMsg(err.message || "Cognitive synthesis failed. Please try again.");
    } finally {
      setIsQuerying(false);
    }
  };

  // Convert markdown links [Title](url) to Apple-style interactive pills
  const renderFormattedText = (text: string) => {
    const lines = text.split("\n");
    return (
      <div className="space-y-3 text-[14.5px] leading-relaxed text-slate-200">
        {lines.map((line, idx) => {
          const trimmed = line.trim();
          if (!trimmed) return <div key={idx} className="h-2" />;

          if (trimmed.startsWith("### ")) {
            return (
              <h4 key={idx} className="text-[16px] font-bold text-teal-300 pt-4 pb-1 border-b border-white/[0.08] flex items-center gap-2 tracking-tight">
                <span className="w-2 h-2 rounded-full bg-teal-400 shadow-[0_0_8px_rgba(45,212,191,0.8)]" />
                <span>{trimmed.replace(/^###\s+/, "")}</span>
              </h4>
            );
          }

          if (trimmed.startsWith("## ")) {
            return (
              <h3 key={idx} className="text-lg font-bold text-white pt-5 pb-1.5 border-b border-white/[0.12] tracking-tight">
                {trimmed.replace(/^##\s+/, "")}
              </h3>
            );
          }

          if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
            return (
              <div key={idx} className="flex items-start gap-2.5 pl-2">
                <span className="text-teal-400 mt-1.5 font-bold">•</span>
                <span className="flex-1">{parseLinks(trimmed.replace(/^[-*]\s+/, ""))}</span>
              </div>
            );
          }

          return <p key={idx}>{parseLinks(trimmed)}</p>;
        })}
      </div>
    );
  };

  const parseLinks = (content: string) => {
    const regex = /\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\)/g;
    const parts = [];
    let lastIdx = 0;
    let match;

    while ((match = regex.exec(content)) !== null) {
      if (match.index > lastIdx) {
        parts.push(content.substring(lastIdx, match.index));
      }
      const label = match[1];
      const url = match[2];
      const isYoutube = url.includes("youtube.com") || url.includes("youtu.be");

      parts.push(
        <a
          key={match.index}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 px-3 py-1 my-0.5 rounded-full bg-white/[0.06] hover:bg-white/[0.12] text-teal-300 hover:text-white border border-teal-400/30 hover:border-teal-400/60 transition-all text-xs font-semibold shadow-[0_2px_8px_rgba(0,0,0,0.3)] select-none"
        >
          {isYoutube ? <Play className="w-3 h-3 fill-red-500 text-red-500" /> : <ExternalLink className="w-3 h-3 text-teal-300" />}
          <span>{label}</span>
        </a>
      );
      lastIdx = match.index + match[0].length;
    }

    if (lastIdx < content.length) {
      parts.push(content.substring(lastIdx));
    }

    return parts.length > 0 ? parts : content;
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-black text-white flex items-center gap-2.5 tracking-tight">
          <BookOpen className="w-7 h-7 text-teal-400" />
          <span>Cognitive Tutor &amp; 3-Phase Mastery Roadmap</span>
        </h2>
        <p className="text-sm text-slate-400 mt-1.5 font-normal">
          Grounded knowledge synthesis with evidence inspections and structured study milestones.
        </p>
      </div>

      {/* Query Bar */}
      <form onSubmit={handleSynthesize} className="apple-glass-card rounded-[26px] p-4 sm:p-5 flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-4 top-3.5 w-4 h-4 text-slate-400" />
          <input
            type="text"
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            placeholder="e.g. prepare for Agentic AI, backpropagation, transformers..."
            className="w-full apple-glass-input pl-11 pr-4 py-3.5 text-sm placeholder:text-slate-400 focus:outline-none"
          />
        </div>

        <select
          value={selectedTopic}
          onChange={(e) => setSelectedTopic(e.target.value)}
          className="px-4 py-3.5 rounded-[18px] bg-slate-900/80 border border-white/10 text-sm text-slate-200 focus:outline-none focus:border-teal-400/60"
        >
          <option value="All Topics">All Topics</option>
          {topics.map((t, idx) => (
            <option key={idx} value={t}>{t}</option>
          ))}
        </select>

        <button
          type="submit"
          disabled={isQuerying}
          className="apple-button-primary px-7 py-3.5 rounded-2xl font-bold text-white text-sm flex items-center justify-center gap-2 whitespace-nowrap disabled:opacity-50 select-none"
        >
          {isQuerying ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span>Synthesizing...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>Synthesize</span>
            </>
          )}
        </button>
      </form>

      {/* Live Neural Pulsing Synthesis Skeleton */}
      {isQuerying && (
        <div className="apple-glass-card rounded-[26px] p-8 space-y-4">
          <div className="flex items-center gap-3.5">
            <div className="w-11 h-11 rounded-2xl bg-teal-500/20 text-teal-300 flex items-center justify-center animate-spin border border-teal-400/30 shadow-[0_0_15px_rgba(20,184,166,0.3)]">
              <RefreshCw className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-white tracking-tight">Synthesizing Cognitive Knowledge</h4>
              <p className="text-xs text-teal-300/90 mt-0.5">
                Scanning vector embeddings • Generating evidence citations • Crafting 3-phase roadmap...
              </p>
            </div>
          </div>
          <div className="w-full bg-slate-900/80 rounded-full h-2 overflow-hidden border border-white/5">
            <div className="bg-gradient-to-r from-teal-400 via-cyan-400 to-indigo-500 h-full w-3/4 animate-pulse shadow-[0_0_12px_rgba(45,212,191,0.5)]" />
          </div>
        </div>
      )}

      {/* Error Notice */}
      {errorMsg && (
        <div className="apple-glass-card rounded-[24px] p-5 border-rose-500/40 bg-rose-950/25 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
            <div className="text-xs text-rose-200">{errorMsg}</div>
          </div>
          <button
            onClick={() => handleSynthesize()}
            className="px-4 py-2 rounded-xl bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 text-xs font-bold transition-all"
          >
            Retry
          </button>
        </div>
      )}

      {/* Grounded Response & Roadmap Cards */}
      {result && (
        <div className="space-y-6">
          {/* Grounded Explanation Card */}
          <div className="apple-glass-card rounded-[28px] p-7 sm:p-9 space-y-5">
            <div className="flex items-center justify-between pb-4 border-b border-white/[0.08]">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-xl bg-teal-500/15 border border-teal-400/30 text-teal-300 flex items-center justify-center shadow-[0_0_12px_rgba(20,184,166,0.25)]">
                  <Sparkles className="w-4 h-4" />
                </div>
                <span className="font-bold text-white text-[15px] tracking-tight">Grounded Cognitive Synthesis</span>
              </div>

              {result.grounded ? (
                <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/15 text-emerald-300 border border-emerald-400/30 flex items-center gap-1.5 shadow-[0_0_12px_rgba(52,211,153,0.2)]">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Grounded in Sources
                </span>
              ) : (
                <span className="px-3 py-1 rounded-full text-xs font-bold bg-amber-500/15 text-amber-300 border border-amber-400/30 flex items-center gap-1.5 shadow-[0_0_12px_rgba(251,191,36,0.2)]">
                  <AlertCircle className="w-3.5 h-3.5" /> Foundational Knowledge
                </span>
              )}
            </div>

            {renderFormattedText(result.answer)}

            {/* Citation Evidence Drawer */}
            {result.sources && result.sources.length > 0 && (
              <div className="pt-4 border-t border-white/[0.08]">
                <button
                  type="button"
                  onClick={() => setShowCitations(!showCitations)}
                  className="text-xs font-bold text-teal-300 hover:text-white flex items-center gap-1.5 transition-colors select-none"
                >
                  <span>{showCitations ? "▾ Hide Citation Evidence" : "▸ Inspect Retrieved Citation Evidence & Match Scores"}</span>
                </button>

                {showCitations && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 mt-4">
                    {result.sources.map((src, i) => (
                      <div key={i} className="p-4 bg-white/[0.03] rounded-2xl border border-white/10 space-y-2 text-xs shadow-inner">
                        <div className="flex items-center justify-between font-bold text-slate-200">
                          <span className="truncate">{src.source}</span>
                          <span className="text-emerald-400 font-mono text-[11px] px-2 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/30">
                            {src.confidence}% Match
                          </span>
                        </div>
                        <div className="text-slate-300 text-[12px] leading-relaxed line-clamp-3 font-normal">
                          {src.text}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* 3-Phase Roadmap Card */}
          {result.roadmap && (
            <div className="apple-glass-card rounded-[28px] p-7 sm:p-9 space-y-5">
              <div className="flex items-center gap-2.5 pb-4 border-b border-white/[0.08]">
                <div className="w-9 h-9 rounded-xl bg-indigo-500/15 border border-indigo-400/30 text-indigo-300 flex items-center justify-center shadow-[0_0_12px_rgba(99,102,241,0.25)]">
                  <Compass className="w-4 h-4" />
                </div>
                <h3 className="font-bold text-white text-[16px] tracking-tight">
                  3-Phase Mastery Study Roadmap &amp; Curated Tutorials
                </h3>
              </div>

              {renderFormattedText(result.roadmap)}

              <div className="pt-5 border-t border-white/[0.08] flex justify-end">
                <button
                  onClick={() => onLaunchQuiz(result.subject || queryText)}
                  className="apple-button-primary px-6 py-3 rounded-2xl font-bold text-white text-xs flex items-center gap-2 select-none"
                >
                  <Award className="w-4 h-4" />
                  <span>Launch Assessment on This Topic</span>
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

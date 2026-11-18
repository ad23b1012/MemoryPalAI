"use client";

import React, { useState } from "react";
import Image from "next/image";
import { Sparkles, Search, ArrowRight, Zap, Award, BookOpen, Brain, CheckCircle2 } from "lucide-react";

interface HeroProps {
  onSearch: (query: string) => void;
  setActiveTab: (tab: string) => void;
  totalDocs?: number;
  totalChunks?: number;
}

export function Hero({ onSearch, setActiveTab, totalDocs = 3, totalChunks = 26 }: HeroProps) {
  const [inputVal, setInputVal] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputVal.trim()) return;
    onSearch(inputVal.trim());
    setActiveTab("chat");
  };

  return (
    <div className="space-y-10 relative">
      {/* Hero Liquid Glass Card (Ultra-Transparent with Specular Caustics) */}
      <div className="liquid-glass-card p-8 sm:p-12 lg:p-16 relative overflow-hidden flex flex-col lg:flex-row items-center justify-between gap-10">
        <div className="relative z-10 max-w-2xl space-y-6">
          {/* Luminous Frosted Pill Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-bold bg-white/[0.10] border border-white/25 backdrop-blur-2xl shadow-[inset_0_1px_1px_rgba(255,255,255,0.45),0_4px_16px_rgba(0,0,0,0.3)]">
            <Sparkles className="w-3.5 h-3.5 text-teal-300 animate-pulse" />
            <span className="bg-gradient-to-r from-teal-200 via-cyan-200 to-indigo-200 bg-clip-text text-transparent">
              Autonomous Cognitive Learning OS
            </span>
          </div>

          {/* Liquid Glass Headline with Vibrant Google/Apple Luster */}
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-[-0.035em] leading-[1.08] text-white drop-shadow-[0_4px_16px_rgba(0,0,0,0.6)]">
            Master Complex Subjects <br className="hidden sm:inline" />
            <span className="bg-gradient-to-r from-teal-300 via-cyan-300 to-indigo-300 bg-clip-text text-transparent">
              With Neural Precision
            </span>
          </h1>

          <p className="text-slate-200 text-sm sm:text-base leading-relaxed max-w-xl font-normal drop-shadow-[0_2px_8px_rgba(0,0,0,0.5)]">
            Transform academic textbooks, recorded lectures, technical research, and web articles 
            into an interconnected cognitive learning system with grounded citation RAG, conceptual 
            quizzes, auto-synthesized revision decks, and <strong className="text-white font-bold underline decoration-teal-400/50 underline-offset-4">SuperMemo SM-2 Spaced Repetition</strong>.
          </p>

          {/* Interactive Omnibar Search in Crystal Translucent Inset */}
          <form onSubmit={handleSubmit} className="pt-2 flex flex-col sm:flex-row gap-3 max-w-xl">
            <div className="relative flex-1">
              <Search className="absolute left-4 top-4 w-4 h-4 text-teal-300" />
              <input
                type="text"
                value={inputVal}
                onChange={(e) => setInputVal(e.target.value)}
                placeholder="Ask any concept from your notes..."
                className="w-full liquid-glass-input pl-11 pr-4 py-3.5 text-sm placeholder:text-slate-400 focus:outline-none font-medium"
              />
            </div>
            <button
              type="submit"
              className="liquid-button-primary px-8 py-3.5 rounded-2xl font-black text-white text-sm flex items-center justify-center gap-2 whitespace-nowrap group select-none shadow-xl"
            >
              <span>Synthesize</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1.5 transition-transform" />
            </button>
          </form>
        </div>

        {/* VisionOS Holographic Floating Liquid Lens */}
        <div className="flex items-center justify-center relative shrink-0 pt-4 lg:pt-0">
          <div className="w-64 h-64 sm:w-72 sm:h-72 liquid-lens flex items-center justify-center relative animate-float">
            {/* Ambient liquid glow radiating behind the emblem */}
            <div className="absolute inset-2 rounded-full bg-gradient-to-tr from-teal-400/40 via-cyan-400/30 to-indigo-500/40 blur-2xl pointer-events-none animate-pulse-slow" />
            
            <Image
              src="/icon.png"
              alt="MemoryPal Agent Emblem"
              width={210}
              height={210}
              className="object-contain drop-shadow-[0_0_40px_rgba(20,184,166,0.7)] relative z-10"
              priority
            />
          </div>
        </div>
      </div>

      {/* 4 Crystal Liquid Glass Telemetry Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 relative z-10">
        {[
          {
            icon: Zap,
            title: "Zero-Latency Vector Store",
            desc: `${totalChunks} document chunks indexed across Pinecone and ChromaDB with 768-dim embeddings.`,
            color: "text-amber-300",
            glow: "rgba(251, 191, 36, 0.4)",
          },
          {
            icon: Brain,
            title: "Multi-Cluster Taxonomies",
            desc: "Extracts semantic entities and knowledge graphs from your uploaded study materials.",
            color: "text-teal-300",
            glow: "rgba(45, 212, 191, 0.4)",
          },
          {
            icon: Award,
            title: "Active Recall Arena",
            desc: "Adaptive multiple-choice testing with real-time scoring and automated revision decks.",
            color: "text-indigo-300",
            glow: "rgba(129, 140, 248, 0.4)",
          },
          {
            icon: CheckCircle2,
            title: "SuperMemo SM-2",
            desc: "Exponential memory decay tracking computing optimal review dates to prevent forgetting.",
            color: "text-emerald-300",
            glow: "rgba(52, 211, 153, 0.4)",
          },
        ].map((feat, i) => {
          const Icon = feat.icon;
          return (
            <div 
              key={i} 
              className="liquid-glass-card rounded-[26px] p-6 space-y-3.5 group cursor-default"
            >
              <div 
                className="w-12 h-12 rounded-2xl bg-white/[0.08] border border-white/25 backdrop-blur-2xl flex items-center justify-center shadow-[inset_0_1.5px_1px_rgba(255,255,255,0.45)] group-hover:scale-110 transition-transform duration-300"
                style={{ boxShadow: `0 0 25px -2px ${feat.glow}, inset 0 1.5px 1px rgba(255, 255, 255, 0.45)` }}
              >
                <Icon className={`w-6 h-6 ${feat.color}`} />
              </div>
              <h4 className="font-extrabold text-white text-[16px] tracking-tight">{feat.title}</h4>
              <p className="text-xs text-slate-200/90 leading-relaxed font-normal">{feat.desc}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

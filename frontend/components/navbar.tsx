"use client";

import React from "react";
import Image from "next/image";
import { Compass, UploadCloud, BookOpen, Award, Share2, BarChart3, Sparkles } from "lucide-react";

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isEngineReady?: boolean;
}

export function Navbar({ activeTab, setActiveTab, isEngineReady = true }: NavbarProps) {
  const navItems = [
    { id: "overview", label: "Overview", icon: Compass },
    { id: "ingest", label: "Ingest Studio", icon: UploadCloud },
    { id: "chat", label: "Cognitive Tutor", icon: BookOpen },
    { id: "quiz", label: "Quiz Arena", icon: Award },
    { id: "graph", label: "Concept Graph", icon: Share2 },
    { id: "analytics", label: "Active Recall", icon: BarChart3 },
  ];

  return (
    <header className="sticky top-0 z-50 px-4 sm:px-8 pt-4 pb-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
        {/* Brand Logo & Name with Liquid Specular Badge */}
        <div 
          className="flex items-center gap-3.5 cursor-pointer select-none group"
          onClick={() => setActiveTab("overview")}
        >
          <div className="relative w-11 h-11 rounded-2xl flex items-center justify-center bg-white/[0.08] border border-white/20 backdrop-blur-2xl shadow-[inset_0_1.5px_1px_rgba(255,255,255,0.4),0_8px_20px_rgba(0,0,0,0.4)] group-hover:scale-105 group-hover:border-teal-300/60 transition-all duration-300">
            <Image
              src="/icon.png"
              alt="MemoryPal Agent Emblem"
              width={34}
              height={34}
              className="object-contain drop-shadow-[0_0_16px_rgba(20,184,166,0.7)]"
              priority
            />
          </div>
          <div>
            <div className="font-black text-lg tracking-tight flex items-center gap-2">
              <span className="text-white drop-shadow-[0_2px_8px_rgba(0,0,0,0.5)]">
                MemoryPal
              </span>
              <span className="text-[10px] font-black tracking-wider text-teal-200 uppercase font-mono px-2.5 py-0.5 rounded-full bg-teal-400/25 border border-teal-300/40 shadow-[0_0_15px_rgba(20,184,166,0.4),inset_0_1px_1px_rgba(255,255,255,0.4)]">
                AGENT
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-white/[0.08] text-slate-300 border border-white/15 hidden sm:inline shadow-inner">
                v2.0
              </span>
            </div>
            <div className="text-[11px] text-teal-200/70 font-medium tracking-normal hidden sm:block">
              Autonomous Cognitive Learning OS
            </div>
          </div>
        </div>

        {/* VisionOS Floating Liquid Segmented Dock */}
        <nav className="hidden lg:flex items-center gap-1 p-1.5 liquid-glass-dock shadow-2xl">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`px-4 py-2 rounded-full transition-all duration-200 flex items-center gap-2 text-xs font-bold select-none ${
                  active
                    ? "liquid-active-pill"
                    : "text-slate-300 hover:text-white hover:bg-white/[0.08]"
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${active ? "text-teal-200" : "text-slate-400"}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* System Status Telemetry */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2.5 px-3.5 py-1.5 rounded-full bg-white/[0.08] border border-white/20 backdrop-blur-2xl text-xs font-semibold shadow-[inset_0_1px_1px_rgba(255,255,255,0.35),0_4px_16px_rgba(0,0,0,0.3)]">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-80" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
            </span>
            <span className="text-emerald-300 text-[11px] tracking-wide">
              {isEngineReady ? "Neural Core Active" : "Initializing..."}
            </span>
          </div>

          <button
            onClick={() => setActiveTab("ingest")}
            className="hidden sm:flex items-center gap-1.5 px-4 py-1.5 rounded-full bg-teal-400/20 hover:bg-teal-400/30 text-teal-200 border border-teal-300/40 text-xs font-bold transition-all shadow-[0_0_18px_rgba(20,184,166,0.35),inset_0_1px_1px_rgba(255,255,255,0.4)] hover:scale-105 active:scale-95"
          >
            <Sparkles className="w-3.5 h-3.5 text-teal-200" />
            <span>Add Source</span>
          </button>
        </div>
      </div>
    </header>
  );
}

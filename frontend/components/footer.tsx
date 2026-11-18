"use client";

import React from "react";
import Image from "next/image";

export function Footer() {
  return (
    <footer className="border-t border-white/[0.08] py-10 px-4 text-center text-xs text-slate-400 space-y-3 relative z-10">
      <div className="flex items-center justify-center gap-2.5">
        <div className="w-8 h-8 rounded-xl bg-white/[0.05] border border-white/10 flex items-center justify-center backdrop-blur-xl shadow-inner">
          <Image
            src="/icon.png"
            alt="MemoryPal Agent Emblem"
            width={22}
            height={22}
            className="object-contain drop-shadow-[0_0_10px_rgba(34,180,173,0.5)]"
          />
        </div>
        <span className="font-extrabold text-base text-white tracking-tight">MemoryPal</span>
        <span className="text-[10px] font-black text-teal-300 uppercase font-mono px-2 py-0.5 rounded-full bg-teal-500/15 border border-teal-400/30">
          AGENT
        </span>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-white/[0.06] text-slate-400 border border-white/10">
          v2.0
        </span>
      </div>
      <div className="font-medium text-slate-400">
        Autonomous Cognitive Learning OS • Powered by Gemini 3.8 Flash, Pinecone &amp; SuperMemo SM-2
      </div>
      <div className="text-[11px] text-slate-500">
        Engineered with Apple VisionOS liquid glass aesthetics, zero-latency vector retrieval, and adaptive spaced repetition.
      </div>
    </footer>
  );
}

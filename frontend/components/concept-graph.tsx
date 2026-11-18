"use client";

import React, { useState, useEffect } from "react";
import { Share2, RefreshCw, Layers, Compass, Tag } from "lucide-react";
import { GraphResponse, fetchGraph } from "@/lib/api";

export function ConceptGraph() {
  const [graphData, setGraphData] = useState<GraphResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const loadGraph = async () => {
    setLoading(true);
    try {
      const data = await fetchGraph();
      setGraphData(data);
    } catch (e: any) {
      console.warn("Graph load error:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadGraph();
  }, []);

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-black text-white flex items-center gap-2.5 tracking-tight">
            <Share2 className="w-7 h-7 text-teal-400" />
            <span>Interactive Concept Knowledge Graph</span>
          </h2>
          <p className="text-sm text-slate-400 mt-1.5 font-normal">
            Entity taxonomy mapping topics, core concepts, and semantic tags extracted from your documents.
          </p>
        </div>

        <button
          onClick={loadGraph}
          disabled={loading}
          className="px-4 py-2 rounded-2xl bg-white/[0.05] hover:bg-white/[0.1] text-slate-200 text-xs font-semibold border border-white/10 flex items-center gap-2 backdrop-blur-xl transition-all shadow-[inset_0_1px_1px_rgba(255,255,255,0.2)]"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          <span>Reload Graph</span>
        </button>
      </div>

      <div className="apple-glass-card rounded-[26px] p-7 space-y-6">
        {/* Entity Legend */}
        <div className="flex flex-wrap gap-4 text-xs font-semibold pb-4 border-b border-white/[0.08]">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-teal-500/10 border border-teal-400/20">
            <span className="w-2.5 h-2.5 rounded-full bg-teal-400 shadow-[0_0_8px_rgba(45,212,191,0.8)]" />
            <span className="text-teal-200">Subject / Category</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-400/20">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
            <span className="text-cyan-200">Core Concept</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-400/20">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-400 shadow-[0_0_8px_rgba(99,102,241,0.8)]" />
            <span className="text-indigo-200">Semantic Tag</span>
          </div>
        </div>

        {/* Visual Entity Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3.5">
          {graphData && graphData.nodes && graphData.nodes.length > 0 ? (
            graphData.nodes.map((node) => (
              <div
                key={node.id}
                className="p-4 rounded-2xl bg-white/[0.03] border border-white/10 hover:border-teal-400/50 hover:bg-white/[0.06] transition-all flex items-center gap-3.5 group cursor-default shadow-inner"
              >
                <div className="w-9 h-9 rounded-xl bg-teal-500/15 border border-teal-400/30 text-teal-300 flex items-center justify-center font-bold text-xs shrink-0 group-hover:scale-105 transition-transform">
                  {node.label.charAt(0)}
                </div>
                <div className="truncate">
                  <div className="font-bold text-xs text-white truncate tracking-tight">{node.label}</div>
                  <div className="text-[11px] text-slate-400 capitalize">{node.group || "Entity"}</div>
                </div>
              </div>
            ))
          ) : (
            <div className="col-span-full py-12 text-center text-slate-500 text-xs">
              No graph entities extracted yet. Upload learning materials in Ingest Studio to build your taxonomy.
            </div>
          )}
        </div>

        {/* Semantic Edges Section */}
        {graphData && graphData.edges && graphData.edges.length > 0 && (
          <div className="pt-4 border-t border-white/[0.08] space-y-3">
            <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
              Semantic Associations ({graphData.edges.length} Connections)
            </h4>
            <div className="flex flex-wrap gap-2">
              {graphData.edges.slice(0, 16).map((edge, idx) => (
                <span
                  key={idx}
                  className="px-3 py-1.5 rounded-full text-xs font-medium bg-white/[0.04] border border-white/10 text-slate-300 flex items-center gap-1.5"
                >
                  <span className="text-teal-300 font-semibold">{edge.from}</span>
                  <span className="text-slate-500 text-[10px]">→</span>
                  <span className="text-indigo-300 font-semibold">{edge.to}</span>
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

"use client";

import React, { useState, useEffect } from "react";
import { BarChart3, Clock, CheckCircle2, RefreshCw, Zap } from "lucide-react";
import { AnalyticsResponse, fetchAnalytics } from "@/lib/api";

export function ActiveRecall() {
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const loadData = async () => {
    setLoading(true);
    try {
      const data = await fetchAnalytics();
      setAnalytics(data);
    } catch (e) {
      console.warn("Analytics load error:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-black text-white flex items-center gap-2.5 tracking-tight">
            <BarChart3 className="w-7 h-7 text-teal-400" />
            <span>Active Recall &amp; SuperMemo SM-2 Engine</span>
          </h2>
          <p className="text-sm text-slate-400 mt-1.5 font-normal">
            Memory retention curves and automated spaced repetition intervals based on your quiz evaluations.
          </p>
        </div>

        <button
          onClick={loadData}
          disabled={loading}
          className="px-4 py-2 rounded-2xl bg-white/[0.05] hover:bg-white/[0.1] text-slate-200 text-xs font-semibold border border-white/10 flex items-center gap-2 backdrop-blur-xl transition-all shadow-[inset_0_1px_1px_rgba(255,255,255,0.2)]"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          <span>Refresh Metrics</span>
        </button>
      </div>

      {/* Top 3 Apple Liquid Glass Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="apple-glass-card rounded-[24px] p-6 space-y-2">
          <div className="text-xs text-slate-400 font-semibold tracking-wide uppercase">Overall Retention Score</div>
          <div className="text-4xl font-black text-teal-300 tracking-tight">
            {analytics ? `${analytics.overall_retention}%` : "100.0%"}
          </div>
          <div className="text-[11px] text-teal-400/80 font-medium">SuperMemo SM-2 E-Factor Tracking</div>
        </div>

        <div className="apple-glass-card rounded-[24px] p-6 space-y-2">
          <div className="text-xs text-slate-400 font-semibold tracking-wide uppercase">Total Assessments Taken</div>
          <div className="text-4xl font-black text-indigo-300 tracking-tight">
            {analytics ? analytics.total_quizzes : 0}
          </div>
          <div className="text-[11px] text-indigo-400/80 font-medium">Grounded Quiz Arena Submissions</div>
        </div>

        <div className="apple-glass-card rounded-[24px] p-6 space-y-2">
          <div className="text-xs text-slate-400 font-semibold tracking-wide uppercase">Topics Due for Review</div>
          <div className="text-4xl font-black text-amber-300 tracking-tight">
            {analytics ? analytics.due_topics.length : 0}
          </div>
          <div className="text-[11px] text-amber-400/80 font-medium">Spaced Interval Notifications</div>
        </div>
      </div>

      {/* Active Recall Topic Mastery Table */}
      <div className="apple-glass-card rounded-[26px] p-6 sm:p-8 space-y-4">
        <div className="flex items-center justify-between pb-3.5 border-b border-white/[0.08]">
          <h3 className="font-bold text-white text-base tracking-tight">Subject Retention &amp; Review Schedule</h3>
          <span className="text-xs font-semibold text-slate-400 px-3 py-1 rounded-full bg-white/[0.05] border border-white/10">
            {analytics?.mastery_list ? analytics.mastery_list.length : 0} Topics Logged
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-white/10 text-slate-400 uppercase tracking-wider text-[11px]">
                <th className="py-3.5 px-4 font-bold">Topic</th>
                <th className="py-3.5 px-4 font-bold">Average Score</th>
                <th className="py-3.5 px-4 font-bold">Quizzes Taken</th>
                <th className="py-3.5 px-4 font-bold">Last Score</th>
                <th className="py-3.5 px-4 font-bold">Next Review</th>
                <th className="py-3.5 px-4 font-bold text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.05] text-slate-300">
              {!analytics || analytics.mastery_list.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-slate-500">
                    No assessments taken yet. Launch Quiz Arena to start memory retention tracking.
                  </td>
                </tr>
              ) : (
                analytics.mastery_list.map((m, idx) => (
                  <tr key={idx} className="hover:bg-white/[0.04] transition-colors">
                    <td className="py-3.5 px-4 font-semibold text-white flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-teal-400" />
                      <span>{m.topic}</span>
                    </td>
                    <td className="py-3.5 px-4 font-mono text-teal-300 font-bold">{m.avg_score}%</td>
                    <td className="py-3.5 px-4 font-mono text-slate-400">{m.attempts}</td>
                    <td className="py-3.5 px-4 font-mono text-slate-300">{m.last_score}%</td>
                    <td className="py-3.5 px-4 text-slate-400">
                      {m.schedule?.next_review ? String(m.schedule.next_review).substring(0, 10) : "Today"}
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      {m.avg_score >= 80 ? (
                        <span className="px-2.5 py-1 rounded-full text-[11px] font-bold bg-emerald-500/15 text-emerald-300 border border-emerald-400/30">
                          Mastered
                        </span>
                      ) : (
                        <span className="px-2.5 py-1 rounded-full text-[11px] font-bold bg-amber-500/15 text-amber-300 border border-amber-400/30">
                          Reinforcing
                        </span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

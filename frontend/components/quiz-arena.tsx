"use client";

import React, { useState } from "react";
import { Award, Zap, CheckCircle2, RefreshCw, AlertCircle, Sparkles } from "lucide-react";
import { QuizGenerateResponse, QuizEvaluateResponse, generateQuiz, evaluateQuiz } from "@/lib/api";

interface QuizArenaProps {
  initialTopic?: string;
  onQuizEvaluated: () => void;
}

export function QuizArena({ initialTopic = "Agentic AI", onQuizEvaluated }: QuizArenaProps) {
  const [topic, setTopic] = useState(initialTopic);
  const [numQuestions, setNumQuestions] = useState(4);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [quizData, setQuizData] = useState<QuizGenerateResponse | null>(null);
  const [selectedAnswers, setSelectedAnswers] = useState<Record<string, string>>({});
  const [evaluation, setEvaluation] = useState<QuizEvaluateResponse | null>(null);

  const handleGenerate = async () => {
    const target = topic.trim() || "Agentic AI";
    setIsGenerating(true);
    setQuizData(null);
    setEvaluation(null);
    setSelectedAnswers({});

    try {
      const data = await generateQuiz(target, "", numQuestions);
      setQuizData(data);
    } catch (e: any) {
      alert("Failed to generate quiz: " + e.message);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSubmit = async () => {
    if (!quizData) return;

    // Map selected option strings to letters (A, B, C, D)
    const letterMap: Record<string, string> = {};
    quizData.questions.forEach((q) => {
      const chosen = selectedAnswers[q.id];
      if (chosen) {
        const idx = q.options.indexOf(chosen);
        letterMap[q.id] = ["A", "B", "C", "D"][idx] || "A";
      } else {
        letterMap[q.id] = "";
      }
    });

    setIsEvaluating(true);
    try {
      const result = await evaluateQuiz(quizData.subject, letterMap, quizData.raw_quiz);
      setEvaluation(result);
      onQuizEvaluated();
    } catch (e: any) {
      alert("Evaluation failed: " + e.message);
    } finally {
      setIsEvaluating(false);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-black text-white flex items-center gap-2.5 tracking-tight">
          <Award className="w-7 h-7 text-teal-400" />
          <span>Adaptive Quiz Arena &amp; Active Recall</span>
        </h2>
        <p className="text-sm text-slate-400 mt-1.5 font-normal">
          Active recall assessment grounded strictly in your uploaded documents with real-time feedback and review.
        </p>
      </div>

      {/* Quiz Controls in Apple Liquid Glass */}
      <div className="apple-glass-card rounded-[26px] p-6 sm:p-8 space-y-5">
        <div className="grid grid-cols-1 sm:grid-cols-12 gap-4">
          <div className="sm:col-span-8">
            <label className="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">
              Topic of Assessment
            </label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g. Agentic AI, Neuroscience, Backpropagation"
              className="w-full apple-glass-input px-4 py-3.5 text-sm placeholder:text-slate-500 focus:outline-none"
            />
          </div>

          <div className="sm:col-span-4">
            <label className="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">
              Questions Count
            </label>
            <select
              value={numQuestions}
              onChange={(e) => setNumQuestions(Number(e.target.value))}
              className="w-full px-4 py-3.5 rounded-[18px] bg-slate-900/80 border border-white/10 text-sm text-slate-200 focus:outline-none focus:border-teal-400/60"
            >
              <option value={3}>3 Questions (Quick Assessment)</option>
              <option value={4}>4 Questions (Standard Review)</option>
              <option value={5}>5 Questions (Comprehensive)</option>
              <option value={8}>8 Questions (Mastery Evaluation)</option>
            </select>
          </div>
        </div>

        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className="w-full apple-button-primary py-4 rounded-2xl font-bold text-white text-sm flex items-center justify-center gap-2 select-none disabled:opacity-50"
        >
          {isGenerating ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span>Synthesizing Document-Grounded Questions...</span>
            </>
          ) : (
            <>
              <Zap className="w-4 h-4 text-teal-200" />
              <span>Generate Interactive Quiz</span>
            </>
          )}
        </button>
      </div>

      {/* Quiz Questions Stream */}
      {quizData && quizData.questions && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="font-black text-xl text-white tracking-tight">
              Topic: <span className="bg-gradient-to-r from-teal-300 to-cyan-300 bg-clip-text text-transparent">{quizData.subject}</span>
            </h3>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-400 font-semibold px-3 py-1 rounded-full bg-white/[0.05] border border-white/10">
                {quizData.questions.length} Questions
              </span>
              {evaluation && (
                <span className={`px-4 py-1.5 rounded-full text-xs font-bold border backdrop-blur-xl shadow-lg ${
                  evaluation.score_pct >= 80 
                    ? "bg-emerald-500/20 text-emerald-300 border-emerald-400/40 shadow-emerald-500/20" 
                    : evaluation.score_pct >= 60 
                    ? "bg-amber-500/20 text-amber-300 border-amber-400/40 shadow-amber-500/20" 
                    : "bg-rose-500/20 text-rose-300 border-rose-400/40 shadow-rose-500/20"
                }`}>
                  Score: {evaluation.score} / {evaluation.total_questions} ({evaluation.score_pct}%)
                </span>
              )}
            </div>
          </div>

          <div className="space-y-5">
            {quizData.questions.map((q) => {
              const qBreakdown = evaluation?.breakdown?.find((b) => b.questionId === q.id);

              return (
                <div
                  key={q.id}
                  className={`apple-glass-card rounded-[26px] p-7 transition-all space-y-5 ${
                    qBreakdown
                      ? qBreakdown.isCorrect
                        ? "border-emerald-500/50 bg-emerald-950/20 shadow-[0_0_30px_rgba(16,185,129,0.15)]"
                        : "border-rose-500/50 bg-rose-950/20 shadow-[0_0_30px_rgba(244,63,94,0.15)]"
                      : ""
                  }`}
                >
                  <div className="font-bold text-[16px] text-white flex items-start justify-between gap-3 tracking-tight">
                    <div className="flex items-start gap-3">
                      <span className="px-2.5 py-1 rounded-xl bg-teal-500/20 text-teal-300 font-mono text-xs font-bold border border-teal-400/30 shrink-0">
                        {q.id}
                      </span>
                      <span className="leading-snug">{q.title}</span>
                    </div>

                    {qBreakdown && (
                      <span className={`px-3 py-1 rounded-full text-xs font-bold shrink-0 ${
                        qBreakdown.isCorrect 
                          ? "bg-emerald-500/20 text-emerald-400 border border-emerald-400/40 shadow-[0_0_12px_rgba(52,211,153,0.3)]" 
                          : "bg-rose-500/20 text-rose-400 border border-rose-400/40 shadow-[0_0_12px_rgba(244,63,94,0.3)]"
                      }`}>
                        {qBreakdown.isCorrect ? "Correct" : "Incorrect"}
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-1">
                    {q.options.map((opt, idx) => {
                      const letter = ["A", "B", "C", "D"][idx];
                      const isSelected = selectedAnswers[q.id] === opt;

                      let btnStyle = "bg-white/[0.03] border-white/10 text-slate-300 hover:border-white/20 hover:bg-white/[0.07]";
                      let circleStyle = "bg-white/[0.08] text-slate-400 border border-white/10";
                      let tagBadge = null;

                      if (qBreakdown) {
                        const isCorrectAnswer = letter === qBreakdown.correctAnswer;
                        const isUserChoice = letter === qBreakdown.userAnswer;

                        if (isCorrectAnswer) {
                          btnStyle = "bg-emerald-500/20 border-emerald-400 text-emerald-100 shadow-[0_0_20px_rgba(16,185,129,0.3),inset_0_1px_1px_rgba(255,255,255,0.4)] font-semibold";
                          circleStyle = "bg-emerald-500 text-white font-black shadow-[0_0_10px_rgba(16,185,129,0.5)]";
                          tagBadge = <span className="text-[11px] font-bold text-emerald-300">✓ Correct Answer</span>;
                        } else if (isUserChoice && !qBreakdown.isCorrect) {
                          btnStyle = "bg-rose-500/20 border-rose-400 text-rose-100 shadow-[0_0_20px_rgba(244,63,94,0.3),inset_0_1px_1px_rgba(255,255,255,0.4)] font-semibold";
                          circleStyle = "bg-rose-500 text-white font-black shadow-[0_0_10px_rgba(244,63,94,0.5)]";
                          tagBadge = <span className="text-[11px] font-bold text-rose-300">✕ Your Choice</span>;
                        } else {
                          btnStyle = "bg-white/[0.02] border-white/[0.05] text-slate-500 opacity-50";
                          circleStyle = "bg-white/[0.05] text-slate-600";
                        }
                      } else if (isSelected) {
                        btnStyle = "bg-teal-500/20 border-teal-400 text-white shadow-[0_0_20px_rgba(20,184,166,0.35),inset_0_1px_1px_rgba(255,255,255,0.3)] font-semibold";
                        circleStyle = "bg-teal-400 text-slate-950 font-black shadow-[0_0_10px_rgba(20,184,166,0.5)]";
                      }

                      return (
                        <button
                          key={idx}
                          type="button"
                          disabled={!!evaluation}
                          onClick={() => setSelectedAnswers({ ...selectedAnswers, [q.id]: opt })}
                          className={`p-4 rounded-2xl border text-left text-sm transition-all duration-200 flex items-center justify-between gap-3 ${btnStyle}`}
                        >
                          <div className="flex items-center gap-3">
                            <div className={`w-7 h-7 rounded-xl flex items-center justify-center font-bold text-xs shrink-0 ${circleStyle}`}>
                              {letter}
                            </div>
                            <span className="flex-1 leading-snug">{opt}</span>
                          </div>
                          {tagBadge}
                        </button>
                      );
                    })}
                  </div>

                  {/* Grounded Explanation Box */}
                  {qBreakdown && (
                    <div className="mt-4 p-4.5 rounded-2xl bg-white/[0.04] border border-teal-400/30 text-xs text-slate-200 space-y-1.5 shadow-inner">
                      <div className="font-bold text-teal-300 flex items-center gap-1.5">
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>Grounded Concept Explanation</span>
                      </div>
                      <p className="leading-relaxed text-slate-300/90 font-normal">{qBreakdown.explanation}</p>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Submission and Evaluation Action Bar */}
          {!evaluation && (
            <div className="apple-glass-card rounded-[24px] p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="text-xs text-slate-400">
                Answered <span className="font-bold text-teal-300">{Object.keys(selectedAnswers).length}</span> of {quizData.questions.length} questions.
              </div>

              <button
                onClick={handleSubmit}
                disabled={isEvaluating || Object.keys(selectedAnswers).length === 0}
                className="apple-button-primary px-8 py-3.5 rounded-2xl font-bold text-white text-sm flex items-center gap-2 select-none disabled:opacity-50"
              >
                {isEvaluating ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Grading Assessment...</span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Submit Assessment &amp; Evaluate</span>
                  </>
                )}
              </button>
            </div>
          )}

          {/* Post-Evaluation SuperMemo SM-2 & Revision Pack Banner */}
          {evaluation && (
            <div className="apple-glass-card rounded-[28px] p-7 sm:p-9 space-y-4">
              <div className="flex items-center gap-2.5 pb-4 border-b border-white/[0.08]">
                <div className="w-9 h-9 rounded-xl bg-teal-500/15 border border-teal-400/30 text-teal-300 flex items-center justify-center">
                  <Award className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="font-bold text-white text-base tracking-tight">SuperMemo SM-2 Interval Calculation</h4>
                  <p className="text-xs text-slate-400">Spaced repetition schedule updated automatically.</p>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-1">
                <div className="p-4 rounded-2xl bg-white/[0.03] border border-white/10 space-y-1">
                  <div className="text-xs text-slate-400">Score Percentage</div>
                  <div className="text-2xl font-black text-teal-300">{evaluation.score_pct}%</div>
                </div>
                <div className="p-4 rounded-2xl bg-white/[0.03] border border-white/10 space-y-1">
                  <div className="text-xs text-slate-400">Next Review Interval</div>
                  <div className="text-2xl font-black text-indigo-300">
                    {evaluation.schedule?.next_review ? `${evaluation.schedule.interval_days || 1} Days` : "Tomorrow"}
                  </div>
                </div>
                <div className="p-4 rounded-2xl bg-white/[0.03] border border-white/10 space-y-1">
                  <div className="text-xs text-slate-400">SM-2 E-Factor</div>
                  <div className="text-2xl font-black text-emerald-300">{evaluation.schedule?.easiness_factor || "2.5"}</div>
                </div>
              </div>

              <div className="pt-4 flex justify-end">
                <button
                  onClick={handleGenerate}
                  className="apple-button-primary px-6 py-3 rounded-2xl font-bold text-white text-xs flex items-center gap-2 select-none"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Take Another Assessment</span>
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

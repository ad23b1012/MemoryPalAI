"use client";

import React, { useState } from "react";
import { UploadCloud, Globe, Play, Trash2, CheckCircle2, RefreshCw, FileText } from "lucide-react";
import { DocumentItem } from "@/lib/api";

interface IngestStudioProps {
  documents: DocumentItem[];
  onRefresh: () => void;
}

export function IngestStudio({ documents, onRefresh }: IngestStudioProps) {
  const [file, setFile] = useState<File | null>(null);
  const [urlInput, setUrlInput] = useState("");
  const [topicInput, setTopicInput] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [audioPreview, setAudioPreview] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      if (selected.name.match(/\.(mp3|wav|m4a)$/i)) {
        setAudioPreview(URL.createObjectURL(selected));
      } else {
        setAudioPreview(null);
      }
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file && !urlInput.trim()) {
      alert("Please choose a file or enter a valid URL.");
      return;
    }

    setIsUploading(true);
    setSuccessMsg(null);

    const formData = new FormData();
    if (file) formData.append("file", file);
    if (urlInput.trim()) formData.append("url", urlInput.trim());
    if (topicInput.trim()) formData.append("topic", topicInput.trim());

    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setSuccessMsg(`Successfully ingested ${data.chunks_count || 1} chunks into knowledge base.`);
        setFile(null);
        setUrlInput("");
        setTopicInput("");
        onRefresh();
      } else {
        alert(data.detail || "Ingestion error");
      }
    } catch (err: any) {
      alert("Ingestion failed: " + err.message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    if (!confirm(`Remove "${docId}" from vector storage?`)) return;
    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(docId)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        onRefresh();
      }
    } catch (e: any) {
      alert("Delete failed: " + e.message);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-black text-white flex items-center gap-2.5 tracking-tight">
          <UploadCloud className="w-7 h-7 text-teal-400" />
          <span>Multimodal Ingest Studio</span>
        </h2>
        <p className="text-sm text-slate-400 mt-1.5 font-normal">
          Upload research PDFs, audio lectures, notes, or web articles to automatically chunk, embed, and index.
        </p>
      </div>

      {/* Ingestion Box */}
      <form onSubmit={handleUpload} className="apple-glass-card rounded-[26px] p-6 sm:p-8 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* File Upload Drop Area */}
          <div className="space-y-2.5">
            <label className="block text-xs font-bold text-slate-300 uppercase tracking-wider">
              Document or Audio Recording
            </label>
            <div className="relative border-2 border-dashed border-white/15 hover:border-teal-400/60 rounded-[22px] p-7 text-center transition-all bg-white/[0.02] hover:bg-white/[0.05] group cursor-pointer shadow-inner">
              <input
                type="file"
                accept=".pdf,.txt,.docx,.mp3,.wav,.m4a"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <div className="flex flex-col items-center justify-center gap-2.5">
                <div className="w-12 h-12 rounded-2xl bg-teal-500/15 border border-teal-400/30 text-teal-300 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <UploadCloud className="w-6 h-6" />
                </div>
                <span className="text-xs font-bold text-slate-200">
                  {file ? file.name : "Drop PDF, Audio, TXT, DOCX here or Browse"}
                </span>
                <span className="text-[11px] text-slate-400">PDF, Markdown, Audio (MP3/WAV/M4A) up to 50MB</span>
              </div>
            </div>

            {audioPreview && (
              <div className="pt-2">
                <audio controls src={audioPreview} className="w-full h-10 rounded-xl" />
              </div>
            )}
          </div>

          {/* Web URL & Topic Metadata */}
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">
                Web Article URL (Optional)
              </label>
              <div className="relative">
                <Globe className="absolute left-4 top-3.5 w-4 h-4 text-slate-400" />
                <input
                  type="url"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  placeholder="https://arxiv.org/html/... or Wikipedia article"
                  className="w-full apple-glass-input pl-11 pr-4 py-3 text-xs placeholder:text-slate-500 focus:outline-none"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">
                Target Subject / Discipline
              </label>
              <input
                type="text"
                value={topicInput}
                onChange={(e) => setTopicInput(e.target.value)}
                placeholder="e.g. Agentic AI, Neuroscience, Quantum Computing"
                className="w-full apple-glass-input px-4 py-3 text-xs placeholder:text-slate-500 focus:outline-none"
              />
            </div>

            <button
              type="submit"
              disabled={isUploading}
              className="w-full apple-button-primary py-3.5 rounded-2xl font-bold text-white text-xs flex items-center justify-center gap-2 select-none disabled:opacity-50"
            >
              {isUploading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Processing &amp; Indexing Vector Space...</span>
                </>
              ) : (
                <>
                  <UploadCloud className="w-4 h-4" />
                  <span>Ingest &amp; Create Vector Embeddings</span>
                </>
              )}
            </button>
          </div>
        </div>

        {successMsg && (
          <div className="p-4 rounded-2xl bg-teal-500/15 border border-teal-400/30 text-xs text-teal-300 flex items-center gap-2.5 shadow-[0_0_15px_rgba(20,184,166,0.2)]">
            <CheckCircle2 className="w-4 h-4 text-teal-400 shrink-0" />
            <span className="font-semibold">{successMsg}</span>
          </div>
        )}
      </form>

      {/* Ingested Documents Inventory Table */}
      <div className="apple-glass-card rounded-[26px] p-6 sm:p-8 space-y-4">
        <div className="flex items-center justify-between pb-3.5 border-b border-white/[0.08]">
          <h3 className="font-bold text-white text-base tracking-tight">Active Knowledge Inventory</h3>
          <span className="text-xs font-semibold text-slate-400 px-3 py-1 rounded-full bg-white/[0.05] border border-white/10">
            {documents.length} Documents Indexed
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-white/10 text-slate-400 uppercase tracking-wider text-[11px]">
                <th className="py-3.5 px-4 font-bold">Document Title</th>
                <th className="py-3.5 px-4 font-bold">Topic</th>
                <th className="py-3.5 px-4 font-bold">Chunks</th>
                <th className="py-3.5 px-4 font-bold">Style / Tone</th>
                <th className="py-3.5 px-4 font-bold text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.05] text-slate-300">
              {documents.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-slate-500">
                    No documents ingested yet. Upload a PDF or add a web URL above.
                  </td>
                </tr>
              ) : (
                documents.map((doc) => (
                  <tr key={doc.document_id} className="hover:bg-white/[0.04] transition-colors">
                    <td className="py-3.5 px-4 font-semibold text-slate-200 flex items-center gap-2.5">
                      <FileText className="w-4 h-4 text-teal-400 shrink-0" />
                      <span className="truncate max-w-[260px]">{doc.document_id}</span>
                    </td>
                    <td className="py-3.5 px-4 text-indigo-300 font-medium">{doc.topic}</td>
                    <td className="py-3.5 px-4 font-mono text-slate-400">{doc.chunks}</td>
                    <td className="py-3.5 px-4 text-slate-400 font-normal">
                      {doc.style || "Descriptive"} / {doc.tone || "Academic"}
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <button
                        onClick={() => handleDelete(doc.document_id)}
                        className="p-2 rounded-xl text-slate-400 hover:text-rose-400 hover:bg-rose-500/15 transition-all"
                        title="Remove Document"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
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

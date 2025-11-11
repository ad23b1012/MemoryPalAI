# app/agents/organizer_agent.py
import os
import time
import json
import re
from typing import Tuple, Dict, Any

from app.services.llm_service import get_llm, generate_with_retry

FAILED_DIR = "/tmp/memorypal_failed_chunks"
RAW_RESP_DIR = "/tmp/memorypal_raw_responses"
os.makedirs(FAILED_DIR, exist_ok=True)
os.makedirs(RAW_RESP_DIR, exist_ok=True)


class OrganizerAgent:
    """
    Extracts entities, relationships, subject, and teaching style safely using Gemini.
    Robust: extracts JSON block, attempts simple repairs, saves raw responses for debugging.
    """

    def __init__(self):
        self.llm = get_llm("gemini-2.5-flash")
        print("✅ OrganizerAgent initialized with Gemini-2.5-flash.")

    # -----------------------------------------------------------------
    def extract_graph_data(self, text: str) -> Dict[str, Any]:
        """Main loop: chunk text, process each chunk, merge partial graphs."""
        print("\n🚀 OrganizerAgent: extracting entities, relationships, and style...")

        if not text or not text.strip():
            print("⚠️ Empty text provided — returning default graph.")
            return self._default_graph()

        chunks = self._split_text(text, max_chunk_size=3000)
        merged_graph = self._default_graph()

        for idx, chunk in enumerate(chunks):
            print(f"🧩 Processing chunk {idx+1}/{len(chunks)} (length={len(chunk)} chars)")
            # small preview without embedding backslash in f-string expressions
            preview = chunk[:200].replace("\n", " ")
            print(f"📖 Preview: {preview}...\n")

            try:
                data = self._process_chunk_with_retries(chunk, idx + 1)
                # if nodes or edges present, merge; otherwise still record subject/style if available
                if data.get("nodes"):
                    merged_graph = self._merge_graphs(merged_graph, data)
                if data.get("subject"):
                    merged_graph["subject"] = data.get("subject")
                if data.get("style"):
                    merged_graph["style"] = data.get("style")
                print(f"✅ Chunk {idx+1} processed. Found {len(data.get('nodes', []))} nodes, {len(data.get('edges', []))} edges.")
            except Exception as e:
                print(f"❌ Chunk {idx+1} processing failed: {e}")
                # continue to next chunk

        # ensure subject/style exist
        merged_graph.setdefault("subject", "Unknown")
        merged_graph.setdefault("style", "Unknown")

        print("✅ Graph extraction complete.")
        return merged_graph

    # -----------------------------------------------------------------
    def _process_chunk_with_retries(self, chunk: str, chunk_no: int) -> Dict[str, Any]:
        """
        Try to obtain valid JSON from the LLM for this chunk.
        Steps:
         1. prompt LLM to return JSON
         2. extract JSON block
         3. attempt json.loads
         4. if fail and nodes exist, attempt a 'repair' pass (asking LLM to return corrected JSON only)
         5. save raw responses and failed chunks for debugging
        """
        base_prompt = f"""
You are MemoryPalAI's Organizer Agent.

Analyze the following text and output a VALID JSON object only (no markdown, no commentary). The JSON must include:
- "nodes": list of objects with "id" and "type"
- "edges": list of objects with "source", "target", "label"
- optionally "subject" and "style"

Example:
{{
  "nodes": [{{"id": "Artificial Intelligence", "type": "Concept"}}],
  "edges": [{{"source": "Artificial Intelligence", "target": "Machine Learning", "label": "includes"}}],
  "subject": "Artificial Intelligence",
  "style": "Example-driven"
}}

Text:
{chunk}
"""

        # 1. initial LLM call
        raw = generate_with_retry(self.llm, base_prompt)
        raw_path = os.path.join(RAW_RESP_DIR, f"chunk_{chunk_no}_raw.txt")
        with open(raw_path, "w", encoding="utf-8") as fh:
            fh.write(raw)
        # 2. try to extract JSON block
        json_block = self._extract_json_block(raw)

        # 3. try load
        try:
            data = json.loads(json_block)
            return self._normalize_graph_dict(data)
        except Exception as e:
            # save failed json and raw
            fail_json_path = os.path.join(FAILED_DIR, f"failed_chunk_{chunk_no}_json_error.txt")
            with open(fail_json_path, "w", encoding="utf-8") as fh:
                fh.write(json_block or "")
            # 4. attempt repair prompt only if nodes present in the raw text (quick heuristic)
            repair_prompt = f"""
The previous response included invalid JSON. Based on this previous raw output:
{raw}

Return a corrected JSON object ONLY (no explanation). Make sure it is valid JSON with keys: nodes, edges, subject, style.
"""
            try:
                repair_raw = generate_with_retry(self.llm, repair_prompt)
                repair_path = os.path.join(RAW_RESP_DIR, f"chunk_{chunk_no}_repair_raw.txt")
                with open(repair_path, "w", encoding="utf-8") as fh:
                    fh.write(repair_raw)

                repair_json_block = self._extract_json_block(repair_raw)
                repaired = json.loads(repair_json_block)
                return self._normalize_graph_dict(repaired)
            except Exception as e2:
                # save repair failure and fallback to minimal structure, but still attempt subject/style extraction
                fail_repair_path = os.path.join(FAILED_DIR, f"failed_chunk_{chunk_no}_repair_exception.txt")
                with open(fail_repair_path, "w", encoding="utf-8") as fh:
                    fh.write(repair_raw if 'repair_raw' in locals() else str(e2))
                # fallback: try to extract subject/style only
                subj, style = self._extract_subject_style_fallback(chunk, chunk_no)
                return {"nodes": [], "edges": [], "subject": subj, "style": style}

    # -----------------------------------------------------------------
    def _extract_json_block(self, text: str) -> str:
        """
        Extract the first balanced JSON object from text. If none, return original text.
        Uses a simple regex to find {...} and returns the largest {...} block.
        """
        if not text:
            return ""
        # Remove markdown code fences first
        cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

        # find first '{' and last '}' to get a block (best-effort)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start:end+1]

        # fallback: attempt regex for smallest {...} block
        m = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
        if m:
            return m.group(0)
        return cleaned

    # -----------------------------------------------------------------
    def _extract_subject_style_fallback(self, chunk: str, chunk_no: int) -> Tuple[str, str]:
        """
        Ask LLM to just give subject and style in a tiny JSON when full graph fails.
        """
        prompt = f"""
Extract the subject (academic topic) and teaching style from the text below.
Return ONLY a JSON object: {{ "subject": "...", "style": "..." }}

Text:
{chunk}
"""
        try:
            raw = generate_with_retry(self.llm, prompt)
            # save for debugging
            p = os.path.join(RAW_RESP_DIR, f"chunk_{chunk_no}_subject_style_raw.txt")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(raw)
            json_block = self._extract_json_block(raw)
            parsed = json.loads(json_block)
            return parsed.get("subject", "Unknown"), parsed.get("style", "Unknown")
        except Exception:
            return "Unknown", "Unknown"

    # -----------------------------------------------------------------
    def _normalize_graph_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize shape so keys exist and are lists.
        Ensure nodes/edges are lists of dicts.
        """
        nodes = data.get("nodes") if isinstance(data.get("nodes"), list) else []
        edges = data.get("edges") if isinstance(data.get("edges"), list) else []
        subject = data.get("subject", "Unknown")
        style = data.get("style", "Unknown")

        # ensure nodes have id/type fields
        normalized_nodes = []
        for n in nodes:
            if isinstance(n, dict) and "id" in n:
                normalized_nodes.append({"id": str(n["id"]), "type": str(n.get("type", "Concept"))})
        normalized_edges = []
        for e in edges:
            if isinstance(e, dict) and all(k in e for k in ("source", "target")):
                normalized_edges.append({
                    "source": str(e["source"]),
                    "target": str(e["target"]),
                    "label": str(e.get("label", "related"))
                })
        return {"nodes": normalized_nodes, "edges": normalized_edges, "subject": subject, "style": style}

    # -----------------------------------------------------------------
    def _split_text(self, text: str, max_chunk_size: int = 3000):
        """Split text into chunks with reasonable overlap."""
        chunks = []
        step = max_chunk_size - 500
        for i in range(0, len(text), step):
            chunks.append(text[i:i + max_chunk_size])
        return chunks

    # -----------------------------------------------------------------
    def _merge_graphs(self, main_graph: Dict[str, Any], new_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Merge nodes/edges without duplication."""
        node_ids = {n["id"] for n in main_graph.get("nodes", [])}
        for n in new_graph.get("nodes", []):
            if n["id"] not in node_ids:
                main_graph["nodes"].append(n)
                node_ids.add(n["id"])
        for e in new_graph.get("edges", []):
            if e not in main_graph.get("edges", []):
                main_graph["edges"].append(e)
        return main_graph

    # -----------------------------------------------------------------
    def _default_graph(self) -> Dict[str, Any]:
        return {"nodes": [{"id": "Document", "type": "Text"}], "edges": [], "subject": "Unknown", "style": "Unknown"}


# ---------------- Test block ----------------
if __name__ == "__main__":
    agent = OrganizerAgent()
    txt = "Artificial Intelligence involves Machine Learning, Deep Learning, and Neural Networks."
    graph = agent.extract_graph_data(txt)
    print("\n✅ Extracted Graph:\n", json.dumps(graph, indent=2))

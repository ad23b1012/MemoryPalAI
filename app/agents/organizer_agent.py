# app/agents/organizer_agent.py
import os
import json
import time
import re
import logging
from typing import Dict, Any, List
from app.services.llm_service import get_llm, generate_with_retry
from app.services.style_detector import detect_style_from_text

logger = logging.getLogger(__name__)

# Lazy spaCy loader
_nlp = None


def get_spacy_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except Exception:
        try:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            return _nlp
        except Exception as e:
            logger.warning(f"spaCy 'en_core_web_sm' unavailable ({e}). Using regex entity fallback.")
            return None


class OrganizerAgent:
    """
    Robust organizer: extracts knowledge graph (nodes + edges),
    calls StyleDetector service, and falls back to NLP heuristics if LLM fails.
    Returns graph dict: {nodes: [...], edges: [...], subject: str, style: str, tone: str, tags: [...]}
    """

    def __init__(self):
        self.llm = get_llm()
        logger.info("OrganizerAgent initialized.")

    def extract_graph_data(self, text: str) -> Dict[str, Any]:
        logger.info("OrganizerAgent: extracting entities, relationships, and style...")
        if not text or not text.strip():
            return self._default_graph()

        chunks = self._split_text(text, max_chunk_size=3000)
        merged = self._default_graph()

        # Detect style from the first chunk
        style_info = detect_style_from_text(chunks[0])
        merged.update(style_info)

        llm_available = True
        for idx, chunk in enumerate(chunks[:2]):  # Process up to 2 chunks for speed & density
            if llm_available:
                try:
                    data = self._process_chunk(chunk)
                    if data and (data.get("nodes") or data.get("edges")):
                        merged = self._merge_graphs(merged, data)
                        continue
                except Exception as e:
                    logger.warning(f"Graph Chunk {idx+1} parsing failed: {e}. Switching to NLP fallback.")
                    llm_available = False

            fallback = self._nlp_fallback(chunk)
            merged = self._merge_graphs(merged, fallback)

        merged.setdefault("subject", style_info.get("subject", "General Topic"))
        merged.setdefault("style", style_info.get("style", "Descriptive"))
        merged.setdefault("tone", style_info.get("tone", "Neutral"))
        merged.setdefault("tags", style_info.get("tags", []))

        return merged

    def _process_chunk(self, chunk: str) -> Dict[str, Any]:
        prompt = f"""
You are an AI that extracts a Knowledge Graph from text. Return ONLY valid JSON with no markdown wrapping.
Extract key concept nodes and relationship edges.

Output format:
{{
  "nodes": [{{"id": "Concept A", "type": "Concept"}}, {{"id": "Concept B", "type": "Topic"}}],
  "edges": [{{"source": "Concept A", "target": "Concept B", "label": "relates_to"}}]
}}

Text:
{chunk[:2000]}
"""
        text = generate_with_retry(self.llm, prompt, retries=1)
        if not text or text.startswith("❌") or "⚠️" in text:
            raise ValueError(f"LLM returned error/unavailable: {text[:60]}")
        json_block = self._extract_json_block(text)
        try:
            data = json.loads(json_block)
            data.setdefault("nodes", [])
            data.setdefault("edges", [])
            return data
        except Exception:
            return self._nlp_fallback(chunk, graph_only=True)

    def _extract_json_block(self, text: str) -> str:
        t = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return t[start:end+1]
        return t

    def _nlp_fallback(self, text: str, graph_only: bool = False) -> Dict[str, Any]:
        nlp = get_spacy_nlp()
        nodes = []
        seen = set()

        if nlp:
            try:
                doc = nlp(text[:2500])
                for ent in doc.ents:
                    k = ent.text.strip()
                    if k and k not in seen and len(k) > 2:
                        nodes.append({"id": k, "type": ent.label_})
                        seen.add(k)
                for nc in doc.noun_chunks:
                    k = nc.text.strip()
                    if len(k) > 3 and k not in seen:
                        nodes.append({"id": k, "type": "Concept"})
                        seen.add(k)
            except Exception:
                pass

        # If spaCy was unavailable or didn't find enough nodes, use regex capitalized phrases
        if len(nodes) < 2:
            phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text[:2000])
            for p in phrases:
                p_clean = p.strip()
                if len(p_clean) > 3 and p_clean not in seen:
                    nodes.append({"id": p_clean, "type": "Concept"})
                    seen.add(p_clean)

        edges = []
        node_ids = [n["id"] for n in nodes[:15]]
        for i in range(len(node_ids) - 1):
            edges.append({
                "source": node_ids[i],
                "target": node_ids[i + 1],
                "label": "connected_to"
            })

        if graph_only:
            return {"nodes": nodes[:20], "edges": edges[:25]}

        subject = nodes[0]["id"] if nodes else "General"
        return {"nodes": nodes[:20], "edges": edges[:25], "subject": subject, "style": "Descriptive"}

    def _split_text(self, text: str, max_chunk_size: int = 3000) -> List[str]:
        chunks = []
        step = max_chunk_size - 500
        for i in range(0, len(text), step):
            chunks.append(text[i:i+max_chunk_size])
        return chunks or [text]

    def _merge_graphs(self, main_graph: Dict[str, Any], new_graph: Dict[str, Any]) -> Dict[str, Any]:
        node_ids = {n["id"] for n in main_graph.get("nodes", [])}
        for n in new_graph.get("nodes", []):
            if n.get("id") and n["id"] not in node_ids:
                main_graph["nodes"].append(n)
                node_ids.add(n["id"])

        edge_keys = {(e.get("source"), e.get("target")) for e in main_graph.get("edges", [])}
        for e in new_graph.get("edges", []):
            key = (e.get("source"), e.get("target"))
            if key not in edge_keys and e.get("source") in node_ids and e.get("target") in node_ids:
                main_graph["edges"].append(e)
                edge_keys.add(key)

        return main_graph

    def _default_graph(self) -> Dict[str, Any]:
        return {
            "nodes": [{"id": "Knowledge Base", "type": "Root"}],
            "edges": [],
            "subject": "General",
            "style": "Descriptive",
            "tone": "Neutral",
            "tags": []
        }

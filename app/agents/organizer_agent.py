# app/agents/organizer_agent.py
import os
import json
import time
import re
import spacy
from app.services.llm_service import get_llm, generate_with_retry
# --- 1. IMPORT THE NEW STYLE DETECTOR ---
from app.services.style_detector import detect_style_from_text

# ensure user installed en_core_web_sm: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

class OrganizerAgent:
    """
    Robust organizer: attempts JSON graph extraction via LLM, 
    calls StyleDetector service, and falls back to spaCy.
    Returns graph dict: {nodes: [...], edges: [...], subject: str, style: str, tone: str, tags: [...]}
    """

    def __init__(self):
        self.llm = get_llm("gemini-2.5-flash")
        print("✅ OrganizerAgent initialized with Gemini-2.5-flash.")

    def extract_graph_data(self, text: str):
        print("\n🚀 OrganizerAgent: extracting entities, relationships, and style...")
        if not text or not text.strip():
            return self._default_graph()

        chunks = self._split_text(text, max_chunk_size=3000)
        merged = self._default_graph()

        # --- 2. DETECT STYLE FROM FIRST CHUNK ---
        # Run the style detector on the first chunk to get the overall doc style
        print("🔬 Calling StyleDetector service...")
        style_info = detect_style_from_text(chunks[0])
        merged.update(style_info) # Update the graph with style, tone, tags
        print(f"✅ Style detected: {style_info['style']} (Tone: {style_info['tone']})")

        # create directories for debugging responses
        os.makedirs("/tmp/memorypal_raw_responses", exist_ok=True)
        os.makedirs("/tmp/memorypal_failed_chunks", exist_ok=True)

        for idx, chunk in enumerate(chunks):
            print(f"🧩 Processing graph chunk {idx+1}/{len(chunks)} (length={len(chunk)} chars)")
            try:
                # --- 3. PROCESS CHUNK FOR GRAPH ONLY ---
                data = self._process_chunk(chunk) # This will now only focus on nodes/edges
                if data and (data.get("nodes") or data.get("edges")):
                    merged = self._merge_graphs(merged, data)
                print(f"✅ Graph Chunk {idx+1} processed. Found {len(data.get('nodes', []))} nodes.")
            except Exception as e:
                print(f"⚠️ Graph Chunk {idx+1} parsing failed: {e}")
                # fallback: spaCy
                fallback = self._spacy_fallback(chunk)
                merged = self._merge_graphs(merged, fallback)

        # ensure keys
        merged.setdefault("subject", style_info.get("subject", "Unknown"))
        merged.setdefault("style", style_info.get("style", "Unknown"))
        merged.setdefault("tone", style_info.get("tone", "Neutral"))
        merged.setdefault("tags", style_info.get("tags", []))
        
        print("✅ Graph extraction complete.")
        return merged

    def _process_chunk(self, chunk: str):
        # --- 4. SIMPLIFIED PROMPT (Focus on Graph) ---
        # We removed subject/style from this prompt; the service handles it.
        prompt = f"""
You are an assistant that returns **only** valid JSON. 
Extract key entities (nodes) and relationships (edges) **only** from the text.

Output example:
{{
  "nodes": [{{"id":"Artificial Intelligence","type":"Concept"}}],
  "edges": [{{"source":"Artificial Intelligence","target":"Machine Learning","label":"includes"}}]
}}
Text:
{chunk}
"""
        # call LLM
        text = generate_with_retry(self.llm, prompt)
        # save raw for debugging
        idx = int(time.time() * 1000) % 1000000
        raw_path = f"/tmp/memorypal_raw_responses/chunk_{idx}_raw.txt"
        try:
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

        # try to extract JSON block
        json_block = self._extract_json_block(text)
        try:
            data = json.loads(json_block)
            # normalize keys
            data.setdefault("nodes", [])
            data.setdefault("edges", [])
            return data
        except Exception as e:
            # save failed chunk + response for debugging
            try:
                i = int(time.time() * 1000) % 1000000
                with open(f"/tmp/memorypal_failed_chunks/failed_chunk_{i}_json_error.txt", "w", encoding="utf-8") as f:
                    f.write(f"chunk:\n{chunk}\n\nresponse:\n{text}\n\nerror:\n{e}")
            except Exception:
                pass
            # fallback to spaCy extraction
            return self._spacy_fallback(chunk, graph_only=True) # Only extract graph

    def _extract_json_block(self, text: str) -> str:
        # remove code fences commonly returned by LLMs
        t = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        # find first {...} balanced block — naive but practical
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return t[start:end+1]
        # otherwise return text (likely not valid JSON)
        return t

    def _spacy_fallback(self, text: str, graph_only=False):
        doc = nlp(text)
        nodes = []
        seen = set()
        for ent in doc.ents:
            key = ent.text.strip()
            if key and key not in seen:
                nodes.append({"id": key, "type": ent.label_})
                seen.add(key)
        for nc in doc.noun_chunks:
            key = nc.text.strip()
            if len(key) > 3 and key not in seen:
                nodes.append({"id": key, "type": "Phrase"})
                seen.add(key)
        edges = []
        sentences = list(doc.sents)
        for sent in sentences:
            present = [n["id"] for n in nodes if n["id"] in sent.text]
            for i in range(len(present)):
                for j in range(i+1, len(present)):
                    edges.append({"source": present[i], "target": present[j], "label": "related_to"})
        
        if graph_only:
            return {"nodes": nodes, "edges": edges}

        subject = nodes[0]["id"] if nodes else "Unknown"
        style = "Descriptive"
        return {"nodes": nodes, "edges": edges, "subject": subject, "style": style}

    # --- THIS IS THE MISSING FUNCTION ---
    def _split_text(self, text: str, max_chunk_size: int = 3000):
        chunks = []
        step = max_chunk_size - 500
        for i in range(0, len(text), step):
            chunks.append(text[i:i+max_chunk_size])
        return chunks
    # -------------------------------------

    def _merge_graphs(self, main_graph, new_graph):
        node_ids = {n["id"] for n in main_graph.get("nodes", [])}
        for n in new_graph.get("nodes", []):
            if n.get("id") not in node_ids:
                main_graph["nodes"].append(n)
                node_ids.add(n.get("id"))
        for e in new_graph.get("edges", []):
            if e not in main_graph.get("edges", []):
                main_graph["edges"].append(e)
        
        # Only update style/subject if it's new
        if new_graph.get("subject") and main_graph.get("subject") in [None, "Unknown"]:
            main_graph["subject"] = new_graph.get("subject")
        if new_graph.get("style") and main_graph.get("style") in [None, "Unknown"]:
            main_graph["style"] = new_graph.get("style")
        if new_graph.get("tone") and main_graph.get("tone") in [None, "Unknown"]:
            main_graph["tone"] = new_graph.get("tone")
        if new_graph.get("tags") and not main_graph.get("tags"):
            main_graph["tags"] = new_graph.get("tags")
        return main_graph

    def _default_graph(self):
        # Default now includes all fields from the style detector
        return {"nodes":[{"id":"Document","type":"Text"}],"edges":[], "subject":"Unknown","style":"Unknown", "tone":"Neutral", "tags":[]}


if __name__ == "__main__":
    agent = OrganizerAgent()
    test_text = "Artificial Intelligence involves Machine Learning, Deep Learning, and Neural Networks."
    graph = agent.extract_graph_data(test_text)
    print(json.dumps(graph, indent=2))
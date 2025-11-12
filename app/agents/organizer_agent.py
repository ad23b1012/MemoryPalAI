# app/agents/organizer_agent.py
import os
import json
import time
import re
import spacy
from app.services.llm_service import get_llm, generate_with_retry

# ensure user installed en_core_web_sm: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

class OrganizerAgent:
    """
    Robust organizer: attempts JSON extraction via LLM, falls back to spaCy-based extraction.
    Returns graph dict: {nodes: [...], edges: [...], subject: str, style: str}
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

        # create directories for debugging responses
        os.makedirs("/tmp/memorypal_raw_responses", exist_ok=True)
        os.makedirs("/tmp/memorypal_failed_chunks", exist_ok=True)

        for idx, chunk in enumerate(chunks):
            print(f"🧩 Processing chunk {idx+1}/{len(chunks)} (length={len(chunk)} chars)")
            preview = chunk[:150].replace("\n", " ")
            print(f"📖 Preview: {preview}...\n")
            try:
                data = self._process_chunk(chunk)
                # persist raw responses for debugging (response saved inside _process_chunk)
                if data and (data.get("nodes") or data.get("edges")):
                    merged = self._merge_graphs(merged, data)
                else:
                    # even if empty nodes/edges, still try merging subject/style if present
                    if data:
                        merged["subject"] = data.get("subject", merged.get("subject", "Unknown"))
                        merged["style"] = data.get("style", merged.get("style", "Unknown"))
                print(f"✅ Chunk {idx+1} processed. Found {len(data.get('nodes', []))} nodes.")
            except Exception as e:
                print(f"⚠️ Chunk {idx+1} parsing failed: {e}")
                # fallback: spaCy
                fallback = self._spacy_fallback(chunk)
                merged = self._merge_graphs(merged, fallback)

        # ensure keys
        merged.setdefault("subject", "Unknown")
        merged.setdefault("style", "Unknown")
        print("✅ Graph extraction complete.")
        return merged

    def _process_chunk(self, chunk: str):
        prompt = f"""
You are an assistant that returns **only** valid JSON. Extract nodes, edges, subject, style.
Output example:
{{
  "nodes": [{{"id":"Artificial Intelligence","type":"Concept"}}],
  "edges": [{{"source":"Artificial Intelligence","target":"Machine Learning","label":"includes"}}],
  "subject": "Artificial Intelligence",
  "style": "Example-driven"
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
            data.setdefault("subject", "Unknown")
            data.setdefault("style", "Unknown")
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
            return self._spacy_fallback(chunk)

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

    def _spacy_fallback(self, text: str):
        doc = nlp(text)
        nodes = []
        seen = set()
        # Named entities first
        for ent in doc.ents:
            key = ent.text.strip()
            if key and key not in seen:
                nodes.append({"id": key, "type": ent.label_})
                seen.add(key)
        # noun chunks as concepts
        for nc in doc.noun_chunks:
            key = nc.text.strip()
            if len(key) > 3 and key not in seen:
                nodes.append({"id": key, "type": "Phrase"})
                seen.add(key)
        # create simple edges by co-occurrence inside sentences
        edges = []
        sentences = list(doc.sents)
        for sent in sentences:
            present = [n["id"] for n in nodes if n["id"] in sent.text]
            for i in range(len(present)):
                for j in range(i+1, len(present)):
                    edges.append({"source": present[i], "target": present[j], "label": "related_to"})
        subject = nodes[0]["id"] if nodes else "Unknown"
        style = "Descriptive"
        return {"nodes": nodes, "edges": edges, "subject": subject, "style": style}

    def _split_text(self, text: str, max_chunk_size: int = 3000):
        chunks = []
        step = max_chunk_size - 500
        for i in range(0, len(text), step):
            chunks.append(text[i:i+max_chunk_size])
        return chunks

    def _merge_graphs(self, main_graph, new_graph):
        node_ids = {n["id"] for n in main_graph.get("nodes", [])}
        for n in new_graph.get("nodes", []):
            if n.get("id") not in node_ids:
                main_graph["nodes"].append(n)
                node_ids.add(n.get("id"))
        for e in new_graph.get("edges", []):
            if e not in main_graph.get("edges", []):
                main_graph["edges"].append(e)
        # prefer subject/style from new_graph if main_graph still unknown
        if new_graph.get("subject") and main_graph.get("subject") in [None, "Unknown"]:
            main_graph["subject"] = new_graph.get("subject")
        if new_graph.get("style") and main_graph.get("style") in [None, "Unknown"]:
            main_graph["style"] = new_graph.get("style")
        return main_graph

    def _default_graph(self):
        return {"nodes":[{"id":"Document","type":"Text"}],"edges":[], "subject":"Unknown","style":"Unknown"}


if __name__ == "__main__":
    agent = OrganizerAgent()
    test_text = "Artificial Intelligence involves Machine Learning, Deep Learning, and Neural Networks."
    graph = agent.extract_graph_data(test_text)
    print(json.dumps(graph, indent=2))

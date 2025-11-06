import time
import json
from app.services.llm_service import get_llm, generate_with_retry

class OrganizerAgent:
    """
    Extracts entities and relationships safely using Gemini 2.5 Flash.
    Includes retries, chunking, fallback graphs, and context isolation.
    """

    def __init__(self):
        self.llm = get_llm("gemini-2.5-flash")
        print("✅ OrganizerAgent initialized with Gemini-2.5-flash.")

    def extract_graph_data(self, text: str):
        """Main graph extraction with retry logic."""
        print("\n🚀 OrganizerAgent: extracting entities and relationships...")

        if not text or len(text.strip()) == 0:
            print("⚠️ Empty text provided — returning default graph.")
            return self._default_graph()

        chunks = self._split_text(text, max_chunk_size=3000)
        merged_graph = self._default_graph()

        for idx, chunk in enumerate(chunks):
            print(f"🧩 Processing chunk {idx + 1}/{len(chunks)} (length={len(chunk)} chars)")
            for attempt in range(3):
                try:
                    sub_graph = self._process_chunk(chunk)
                    if sub_graph:
                        merged_graph = self._merge_graphs(merged_graph, sub_graph)
                    break
                except Exception as e:
                    print(f"⚠️ Chunk {idx + 1} attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep((attempt + 1) * 3)
                    else:
                        print(f"❌ Failed to process chunk {idx + 1} after 3 attempts.")
        print("✅ Graph extraction complete.")
        return merged_graph

    def _process_chunk(self, chunk: str):
        """Process a single chunk through Gemini with enforced JSON output."""
        prompt = f"""
        You are MemoryPalAI's Organizer Agent.

        Analyze the provided text and extract structured knowledge strictly in JSON format.
        Identify key entities (nodes) and relationships (edges) **only** from the text.

        Output must be valid JSON like:
        {{
          "nodes": [{{"id": "EntityName", "type": "Type"}}],
          "edges": [{{"source": "EntityName1", "target": "EntityName2", "label": "relationship"}}]
        }}

        Text:
        {chunk}
        """

        try:
            response_text = generate_with_retry(self.llm, prompt)
            return json.loads(response_text)
        except json.JSONDecodeError:
            print("⚠️ Failed to parse JSON, returning empty subgraph.")
            return {"nodes": [], "edges": []}
        except Exception as e:
            print(f"❌ Error during chunk processing: {e}")
            return {"nodes": [], "edges": []}

    def _split_text(self, text: str, max_chunk_size: int = 3000):
        """Split text into manageable overlapping chunks."""
        chunks, step = [], max_chunk_size - 500
        for i in range(0, len(text), step):
            chunks.append(text[i:i + max_chunk_size])
        return chunks

    def _merge_graphs(self, main_graph, new_graph):
        """Merge multiple partial graphs without duplication."""
        node_ids = {node["id"] for node in main_graph["nodes"]}
        for node in new_graph.get("nodes", []):
            if node.get("id") not in node_ids:
                main_graph["nodes"].append(node)
                node_ids.add(node["id"])

        for edge in new_graph.get("edges", []):
            if edge not in main_graph["edges"]:
                main_graph["edges"].append(edge)
        return main_graph

    def _default_graph(self):
        """Return a safe fallback graph if LLM extraction fails."""
        return {
            "nodes": [{"id": "Document", "type": "Text"}],
            "edges": []
        }


if __name__ == "__main__":
    agent = OrganizerAgent()
    test_text = "AI involves Machine Learning and Deep Learning."
    graph = agent.extract_graph_data(test_text)
    print(json.dumps(graph, indent=2))

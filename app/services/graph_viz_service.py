# app/services/graph_viz_service.py
import networkx as nx
from pyvis.network import Network
import os
import json

class GraphVisualizer:
    """
    Visualizes a knowledge graph (nodes + edges) extracted by OrganizerAgent.
    """

    def __init__(self, output_dir: str = "./graph_visuals"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✅ GraphVisualizer initialized (output: {self.output_dir})")

    def create_graph(self, graph_data: dict):
        """
        Builds a NetworkX graph from the JSON structure.
        """
        G = nx.DiGraph()

        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        # Add nodes
        for node in nodes:
            node_id = node.get("id", "Unknown")
            node_type = node.get("type", "Concept")
            G.add_node(node_id, label=node_id, type=node_type)

        # Add edges
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            label = edge.get("label", "")
            if src and tgt:
                G.add_edge(src, tgt, label=label)

        return G

    def visualize(self, graph_data: dict, output_name: str = "knowledge_graph.html") -> str:
        """
        Creates an interactive HTML visualization and saves it locally.
        Returns the full file path to the generated HTML.
        """
        try:
            G = self.create_graph(graph_data)
            net = Network(height="750px", width="100%", directed=True, bgcolor="#222222", font_color="white")

            net.from_nx(G)

            # Customize appearance
            for node in net.nodes:
                node_type = G.nodes[node['id']].get("type", "Concept")
                if node_type.lower() == "person":
                    node["color"] = "#FFA500"  # orange
                elif node_type.lower() == "organization":
                    node["color"] = "#1E90FF"  # blue
                elif node_type.lower() == "concept":
                    node["color"] = "#32CD32"  # green
                else:
                    node["color"] = "#CCCCCC"  # grey

            for edge in net.edges:
                edge["color"] = "#AAAAAA"
                edge["arrows"] = "to"

            output_path = os.path.join(self.output_dir, output_name)
            net.show(output_path)
            print(f"✅ Graph visualization saved at: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            return None


# ---------------- Test ----------------
if __name__ == "__main__":
    # Sample graph data (like OrganizerAgent output)
    sample_graph = {
        "nodes": [
            {"id": "Alan Turing", "type": "Person"},
            {"id": "Turing Machine", "type": "Concept"},
            {"id": "University of Manchester", "type": "Organization"},
        ],
        "edges": [
            {"source": "Alan Turing", "target": "Turing Machine", "label": "created"},
            {"source": "Turing Machine", "target": "University of Manchester", "label": "researched at"},
        ],
    }

    viz = GraphVisualizer()
    html_path = viz.visualize(sample_graph, "turing_graph.html")
    print(f"🌐 Open in browser: {html_path}")

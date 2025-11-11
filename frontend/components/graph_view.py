# components/graph_view.py
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os


def render_knowledge_graph(graph_data):
    """Render knowledge graph interactively using PyVis."""
    net = Network(height="450px", width="100%", bgcolor="#222222", font_color="white")
    net.barnes_hut()

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    for node in nodes:
        net.add_node(node["id"], label=node["id"], title=node.get("type", ""), color="#00BFFF")

    for edge in edges:
        net.add_edge(edge["source"], edge["target"], label=edge["label"], color="#FF6F61")

    tmp_dir = tempfile.mkdtemp()
    html_path = os.path.join(tmp_dir, "graph.html")
    net.save_graph(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html_data = f.read()

    components.html(html_data, height=460, scrolling=True)

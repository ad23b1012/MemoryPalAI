# frontend/components/graph_view.py
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

def render_knowledge_graph(graph_data):
    """Render knowledge graph interactively using PyVis. Safely skip broken edges."""
    net = Network(height="450px", width="100%", bgcolor="#222222", font_color="white")
    net.barnes_hut()

    nodes = graph_data.get("nodes", []) or []
    edges = graph_data.get("edges", []) or []

    node_ids = set()
    for node in nodes:
        nid = node.get("id") or str(node)
        node_ids.add(nid)
        net.add_node(nid, label=nid, title=node.get("type", ""), color="#00BFFF")

    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if not src or not tgt:
            continue
        if src not in node_ids or tgt not in node_ids:
            # skip edges referencing missing nodes
            continue
        net.add_edge(src, tgt, label=edge.get("label", ""), color="#FF6F61")

    tmp_dir = tempfile.mkdtemp()
    html_path = os.path.join(tmp_dir, "graph.html")
    net.save_graph(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html_data = f.read()

    components.html(html_data, height=460, scrolling=True)

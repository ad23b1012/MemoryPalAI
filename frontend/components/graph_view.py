# frontend/components/graph_view.py
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

def render_knowledge_graph(graph_data):
    """Render knowledge graph interactively using PyVis with safety checks."""
    net = Network(height="450px", width="100%", bgcolor="#222222", font_color="white")
    try:
        net.barnes_hut()
    except Exception:
        pass

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    node_ids = set()
    for node in nodes:
        node_id = str(node.get("id") or node.get("name") or "")
        if not node_id:
            continue
        node_ids.add(node_id)
        label = node.get("label", node_id)
        title = node.get("type", "")
        # add node safely
        try:
            net.add_node(node_id, label=label, title=title)
        except Exception:
            # fallback: skip problematic node
            continue

    # Add edges only if both endpoints exist
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        lbl = edge.get("label", "")
        if not src or not tgt:
            continue
        if src not in node_ids or tgt not in node_ids:
            # skip edges that reference non-existent nodes
            continue
        try:
            net.add_edge(src, tgt, label=lbl)
        except Exception:
            continue

    tmp_dir = tempfile.mkdtemp()
    html_path = os.path.join(tmp_dir, "graph.html")
    net.save_graph(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html_data = f.read()

    components.html(html_data, height=460, scrolling=True)

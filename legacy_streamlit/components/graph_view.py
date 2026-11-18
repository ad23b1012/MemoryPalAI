# frontend/components/graph_view.py
import os
import tempfile
import streamlit.components.v1 as components
from pyvis.network import Network


def render_knowledge_graph(graph_data: dict, height: str = "480px"):
    """
    Renders an interactive, futuristic concept network with physics simulation
    and color-coded node types.
    """
    if not graph_data or not graph_data.get("nodes"):
        return

    net = Network(
        height=height,
        width="100%",
        bgcolor="#0b0f19",
        font_color="#f1f5f9",
        directed=True,
    )

    # Physics configuration for smooth organic repulsion
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.015,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 45,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 120}
      },
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "shadow": {"enabled": true, "color": "rgba(0,0,0,0.5)", "size": 10}
      },
      "edges": {
        "color": {"color": "#475569", "highlight": "#6366f1"},
        "smooth": {"type": "continuous"},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 150,
        "zoomView": true
      }
    }
    """)

    nodes = graph_data.get("nodes", []) or []
    edges = graph_data.get("edges", []) or []

    # Palette mapped by entity type
    color_map = {
        "Topic": {"background": "#8B5CF6", "border": "#A78BFA"},     # Violet
        "Concept": {"background": "#06B6D4", "border": "#38BDF8"},   # Cyan
        "Tag": {"background": "#10B981", "border": "#34D399"},       # Emerald
        "Phrase": {"background": "#F59E0B", "border": "#FBBF24"},    # Amber
        "Root": {"background": "#EC4899", "border": "#F472B6"},      # Pink
    }
    default_color = {"background": "#3B82F6", "border": "#60A5FA"}

    node_ids = set()
    for node in nodes:
        nid = str(node.get("id", "")).strip()
        if not nid:
            continue
        ntype = str(node.get("type", "Concept"))
        palette = color_map.get(ntype, default_color)

        node_ids.add(nid)
        size = 28 if ntype in ("Topic", "Root") else 20
        net.add_node(
            nid,
            label=nid,
            title=f"<b>{nid}</b><br><i>Type: {ntype}</i>",
            color=palette,
            size=size,
            font={"size": 13, "color": "#f8fafc", "face": "system-ui, sans-serif"},
        )

    for edge in edges:
        src = str(edge.get("source", "")).strip()
        tgt = str(edge.get("target", "")).strip()
        lbl = str(edge.get("label", ""))

        if src in node_ids and tgt in node_ids and src != tgt:
            net.add_edge(src, tgt, title=lbl, label=lbl, color="#475569")

    # Generate isolated HTML
    tmp_dir = tempfile.mkdtemp()
    html_path = os.path.join(tmp_dir, "graph.html")
    net.save_graph(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html_data = f.read()

    components.html(html_data, height=500, scrolling=False)

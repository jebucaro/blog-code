import logging
import webbrowser
from pathlib import Path

from pyvis.network import Network

from nodus.models import KnowledgeGraph
from nodus.settings import Settings

logger = logging.getLogger(__name__)

# Semantic color per controlled node type, tuned per background.
# Keys must exactly match NODE_TYPES in nodus.models (guarded by tests).
NODE_TYPE_COLORS = {
    "dark": {
        "person": "#60a5fa",
        "organization": "#fbbf24",
        "location": "#34d399",
        "event": "#f87171",
        "concept": "#a78bfa",
        "product": "#f472b6",
        "technology": "#22d3ee",
        "date": "#94a3b8",
        "other": "#9ca3af",
    },
    "light": {
        "person": "#2563eb",
        "organization": "#d97706",
        "location": "#059669",
        "event": "#dc2626",
        "concept": "#7c3aed",
        "product": "#db2777",
        "technology": "#0891b2",
        "date": "#64748b",
        "other": "#6b7280",
    },
}

THEMES = {
    "dark": {
        "background": "#222222",
        "font_color": "white",
        "default_node_color": "#7f7f7f",
        "edge_font_stroke": "#000000"
    },
    "light": {
        "background": "#ffffff",
        "font_color": "black",
        "default_node_color": "#7f7f7f",
        "edge_font_stroke": "#ffffff"
    }
}


class GraphVisualizer:
    """Create interactive graph visualizations with PyVis"""

    def __init__(self, settings: Settings | None = None, theme: str | None = None):
        self.settings = settings or Settings()
        theme_key = theme if theme is not None else self.settings.viz_theme
        self.theme_name = theme_key if theme_key in THEMES else "dark"
        self.theme = THEMES[self.theme_name]
        self.type_colors = NODE_TYPE_COLORS[self.theme_name]

        self.viz_height = "100vh"
        self.viz_width = "100%"
        self.show_isolated = False
        self.physics_enabled = True

    def _get_color_for_node_type(self, node_type: str) -> str:
        """Look up the semantic color for a node type, falling back to 'other'."""
        return self.type_colors.get(node_type, self.type_colors["other"])

    def _format_relationship_tooltip(self, rel_type: str, source_label: str, target_label: str) -> str:
        """Create a readable tooltip for relationships"""
        return f"{source_label} → {rel_type.replace('_', ' ').title()} → {target_label}"

    def _build_network(
        self,
        graph: KnowledgeGraph,
        placeholder_ids: set[str] | None = None,
    ) -> tuple[Network, set[str]]:
        """Build a PyVis Network from a KnowledgeGraph; returns (network, rendered node types)."""
        placeholder_ids = placeholder_ids or set()

        net = Network(
            height=self.viz_height,
            width=self.viz_width,
            directed=True,
            notebook=False,
            bgcolor=self.theme["background"],
            font_color=self.theme["font_color"]
        )

        node_dict = {node.id: node for node in graph.nodes}

        valid_edges = []
        valid_node_ids = set()
        for rel in graph.relationships:
            if (rel.source_node_id != rel.target_node_id and
                    rel.source_node_id in node_dict and
                    rel.target_node_id in node_dict):
                valid_edges.append(rel)
                valid_node_ids.update([rel.source_node_id, rel.target_node_id])

        excluded_edges = len(graph.relationships) - len(valid_edges)
        if excluded_edges:
            logger.warning(
                "Excluded %d relationship(s) with dangling or self-referencing endpoints",
                excluded_edges,
            )
        hidden_nodes = [node.id for node in graph.nodes if node.id not in valid_node_ids]
        if hidden_nodes and not self.show_isolated:
            logger.info("Hiding %d isolated node(s): %s", len(hidden_nodes), hidden_nodes)

        types_present: set[str] = set()
        for node in graph.nodes:
            if self.show_isolated or node.id in valid_node_ids:
                node_type = node.type.lower()
                color = self._get_color_for_node_type(node_type)

                display_label = node.label if node.label else ' '.join(
                    word.capitalize() for word in node.id.split('_'))
                tooltip = f"{display_label}\nType: {node.type}\nID: {node.id}"

                node_options = {
                    "label": display_label,
                    "title": tooltip,
                    "color": color,
                    "size": 30,
                    "font": {"size": 14, "color": self.theme["font_color"]},
                }
                if node.id in placeholder_ids:
                    node_options["shapeProperties"] = {"borderDashes": [5, 5]}
                    node_options["title"] += "\n(inferred placeholder)"

                try:
                    net.add_node(node.id, **node_options)
                    types_present.add(node_type)
                except Exception:
                    continue

        for rel in valid_edges:
            try:
                source_node = node_dict.get(rel.source_node_id)
                target_node = node_dict.get(rel.target_node_id)

                if source_node and target_node:
                    source_label = source_node.label if source_node.label else ' '.join(
                        word.capitalize() for word in source_node.id.split('_'))
                    target_label = target_node.label if target_node.label else ' '.join(
                        word.capitalize() for word in target_node.id.split('_'))

                    edge_tooltip = self._format_relationship_tooltip(
                        rel.type,
                        source_label,
                        target_label
                    )

                    display_label = rel.type.replace('_', ' ').title()

                    net.add_edge(
                        rel.source_node_id,
                        rel.target_node_id,
                        label=display_label,
                        title=edge_tooltip,
                        width=2,
                        font={"size": 12, "color": self.theme["font_color"], "strokeWidth": 2,
                              "strokeColor": self.theme["edge_font_stroke"]},
                        arrows={"to": {"enabled": True, "scaleFactor": 1.2}}
                    )
            except Exception:
                continue

        if self.physics_enabled:
            net.set_options("""
            {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -100,
                        "centralGravity": 0.005,
                        "springLength": 150,
                        "springConstant": 0.08,
                        "damping": 0.4
                    },
                    "minVelocity": 0.75,
                    "solver": "forceAtlas2Based",
                    "stabilization": {"iterations": 100, "fit": false}
                },
                "interaction": {
                    "hover": true,
                    "tooltipDelay": 300,
                    "hideEdgesOnDrag": true
                },
                "nodes": {
                    "borderWidth": 2,
                    "borderWidthSelected": 4
                },
                "edges": {
                    "smooth": {
                        "type": "dynamic",
                        "roundness": 1
                    }
                }
            }
            """)

        return net, types_present

    def _legend_html(self, types_present: set[str]) -> str:
        """Fixed-position legend overlay for the node types actually rendered."""
        if not types_present:
            return ""
        if self.theme_name == "dark":
            panel_bg, panel_border = "rgba(34,34,34,0.88)", "#444444"
        else:
            panel_bg, panel_border = "rgba(255,255,255,0.92)", "#cccccc"
        items = "".join(
            '<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
            f'<span style="width:10px;height:10px;border-radius:50%;'
            f'background:{self._get_color_for_node_type(t)};display:inline-block;"></span>'
            f'<span>{t}</span></div>'
            for t in sorted(types_present)
        )
        return (
            f'<div id="nodus-legend" style="position:fixed;top:12px;right:12px;z-index:10;'
            f'background:{panel_bg};border:1px solid {panel_border};border-radius:8px;'
            f'padding:8px 12px;font-family:sans-serif;font-size:12px;'
            f'color:{self.theme["font_color"]};">{items}</div>'
        )

    def generate_html(
        self,
        graph: KnowledgeGraph,
        placeholder_ids: set[str] | None = None,
    ) -> str:
        """Generate HTML visualization as a string (in-memory, no file I/O)."""
        net, types_present = self._build_network(graph, placeholder_ids)
        html_content = net.generate_html()

        # Override PyVis defaults: make the canvas fill the entire iframe.
        # PyVis generates #mynetwork with height:600px and float:left inside a
        # Bootstrap card, with empty <center><h1></h1></center> elements that
        # consume vertical space. This forces everything to 100vh so that
        # network.fit() centres within the full available area.
        css_override = (
            "<style>"
            "html,body{height:100%;margin:0;padding:0;overflow:hidden;}"
            "center,h1{display:none;}"
            ".card{height:100vh;border:0!important;margin:0!important;padding:0!important;}"
            "#mynetwork{width:100%!important;height:100vh!important;border:0!important;float:none!important;}"
            "</style>"
        )
        html_content = html_content.replace("</head>", css_override + "</head>")

        fit_script = (
            'var _physicsOff = false;\n'
            'var _fitOpts = { animation: { duration: 500, easingFunction: "easeInOutQuad" } };\n'
            'var _fitTimer = null;\n'
            'function _scheduleFit() {\n'
            '    if (!_physicsOff) return;\n'
            '    if (_fitTimer !== null) clearTimeout(_fitTimer);\n'
            '    _fitTimer = setTimeout(function() {\n'
            '        _fitTimer = null;\n'
            '        requestAnimationFrame(function() { network.fit(_fitOpts); });\n'
            '    }, 200);\n'
            '}\n'
            'network.once("stabilizationIterationsDone", function() {\n'
            '    network.setOptions({ physics: { enabled: false } });\n'
            '    _physicsOff = true;\n'
            '    requestAnimationFrame(function() { network.fit(_fitOpts); });\n'
            '    setTimeout(function() { _scheduleFit(); }, 250);\n'
            '    setTimeout(function() { _scheduleFit(); }, 700);\n'
            '});\n'
            'var _lastW = 0, _lastH = 0;\n'
            'var _ro = new ResizeObserver(function(entries) {\n'
            '    for (var e of entries) {\n'
            '        var w = e.contentRect.width, h = e.contentRect.height;\n'
            '        if (w > 0 && h > 0 && (w !== _lastW || h !== _lastH)) {\n'
            '            _lastW = w; _lastH = h;\n'
            '            _scheduleFit();\n'
            '        }\n'
            '    }\n'
            '});\n'
            '_ro.observe(document.getElementById("mynetwork"));\n'
            'setTimeout(function() {\n'
            '    if (!_physicsOff) {\n'
            '        _physicsOff = true;\n'
            '        requestAnimationFrame(function() { network.fit(_fitOpts); });\n'
            '    }\n'
            '}, 2000);\n'
        )
        html_content = html_content.replace(
            "network = new vis.Network(container, data, options);",
            "network = new vis.Network(container, data, options);\n" + fit_script
        )
        html_content = html_content.replace(
            "</body>", self._legend_html(types_present) + "</body>"
        )
        logger.info("Generated HTML visualization in memory")
        return html_content

    def visualize(
            self,
            graph: KnowledgeGraph,
            output_file: str = "output/knowledge_graph.html",
            auto_open: bool = True
    ) -> Path:
        """Generate and save HTML visualization to a file."""
        net, _ = self._build_network(graph)

        output_path = Path(output_file).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output_path))

        logger.info(f"Saved visualization to {output_path}")

        if auto_open:
            try:
                webbrowser.open(f"file://{output_path}")
            except Exception:
                logger.warning("Could not open browser automatically")

        return output_path

    def _get_connected_nodes(self, graph: KnowledgeGraph) -> set[str]:
        """Get IDs of nodes that have relationships"""
        connected = set()
        for rel in graph.relationships:
            connected.add(rel.source_node_id)
            connected.add(rel.target_node_id)
        return connected

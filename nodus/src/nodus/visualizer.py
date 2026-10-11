import hashlib
import json
import logging

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
        "edge_font_stroke": "#000000",
        # Translucent canvas-background color behind node labels, so an edge
        # passing under a label doesn't visually merge with the text while
        # still showing through.
        "label_background": "rgba(34, 34, 34, 0.75)",
    },
    "light": {
        "background": "#ffffff",
        "font_color": "black",
        "default_node_color": "#7f7f7f",
        "edge_font_stroke": "#ffffff",
        "label_background": "rgba(255, 255, 255, 0.75)",
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

    def _get_color_for_relationship_type(self, rel_type: str) -> str:
        """Deterministic color for a relationship type.

        Relationship types are open-ended (the model can coin new ones beyond
        RELATIONSHIP_TYPE_EXAMPLES), so unlike node types there's no fixed
        palette to look up - the hue is derived from a hash of the type name
        instead, keeping the same type the same color across renders.
        """
        digest = hashlib.md5(rel_type.encode("utf-8")).hexdigest()
        hue = int(digest[:8], 16) % 360
        saturation, lightness = (65, 65) if self.theme_name == "dark" else (70, 40)
        return f"hsl({hue}, {saturation}%, {lightness}%)"

    def _format_relationship_tooltip(self, rel_type: str, source_label: str, target_label: str) -> str:
        """Create a readable tooltip for relationships"""
        return f"{source_label} → {rel_type.replace('_', ' ').title()} → {target_label}"

    def _rendered_relationship_types(self, graph: KnowledgeGraph) -> list[str]:
        """Distinct relationship types among edges that will actually render.

        Mirrors the self-loop / dangling-endpoint filtering in _build_network so
        the edge-type filter only lists types that are truly drawn. Returned
        sorted case-insensitively for a stable, readable filter panel.
        """
        node_ids = {node.id for node in graph.nodes}
        types = {
            rel.type
            for rel in graph.relationships
            if (rel.source_node_id != rel.target_node_id
                and rel.source_node_id in node_ids
                and rel.target_node_id in node_ids)
        }
        return sorted(types, key=str.lower)

    def _build_network(
        self,
        graph: KnowledgeGraph,
        placeholder_ids: set[str] | None = None,
    ) -> tuple[Network, set[str]]:
        """Build a PyVis Network from a KnowledgeGraph; returns (network, rendered node types)."""
        placeholder_ids = placeholder_ids or set()

        # NOTE: no font_color kwarg here - pyvis's Network(font_color=...) forces
        # every node's `font` dict to {"color": font_color}, silently discarding
        # per-node font options (including our contrast-aware label color below).
        net = Network(
            height=self.viz_height,
            width=self.viz_width,
            directed=True,
            notebook=False,
            bgcolor=self.theme["background"],
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
                if node.context:
                    tooltip += f"\n{node.context}"

                node_options = {
                    "label": display_label,
                    "title": tooltip,
                    "color": color,
                    "size": 30,
                    "font": {
                        "size": 14,
                        "color": self.theme["font_color"],
                        "background": self.theme["label_background"],
                    },
                }
                if node.id in placeholder_ids:
                    node_options["shapeProperties"] = {"borderDashes": [5, 5]}
                    node_options["title"] += "\n(inferred placeholder)"

                try:
                    net.add_node(node.id, **node_options)
                    types_present.add(node_type)
                except Exception:
                    logger.warning("Skipping node %r: failed to add to visualization", node.id, exc_info=True)
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
                    if rel.context:
                        edge_tooltip += f"\n{rel.context}"

                    display_label = rel.type.replace('_', ' ').title()

                    net.add_edge(
                        rel.source_node_id,
                        rel.target_node_id,
                        label=display_label,
                        title=edge_tooltip,
                        width=2,
                        color=self._get_color_for_relationship_type(rel.type),
                        rel_type=rel.type,
                        font={"size": 12, "color": self.theme["font_color"], "strokeWidth": 2,
                              "strokeColor": self.theme["edge_font_stroke"], "align": "top"},
                        arrows={"to": {"enabled": True, "scaleFactor": 1.2}}
                    )
            except Exception:
                logger.warning(
                    "Skipping relationship %r: failed to add to visualization", rel.id, exc_info=True
                )
                continue

        if self.physics_enabled:
            net.set_options("""
            {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -100,
                        "centralGravity": 0.005,
                        "springLength": 200,
                        "springConstant": 0.08,
                        "damping": 0.4,
                        "avoidOverlap": 0.5
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

    def _panel_style(self) -> tuple[str, str]:
        """(background, border) colors for theme-matched overlay panels."""
        if self.theme_name == "dark":
            return "rgba(34,34,34,0.88)", "#444444"
        return "rgba(255,255,255,0.92)", "#cccccc"

    def _legend_html(self, types_present: set[str]) -> str:
        """Fixed-position legend overlay for the node types actually rendered."""
        if not types_present:
            return ""
        panel_bg, panel_border = self._panel_style()
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

    def _edge_filter_html(self, rel_types: list[str]) -> str:
        """Overlay panel + JS to show/hide edges by relationship type.

        Doubles as a color legend: each row's swatch matches the color used
        for that relationship type's edges in the graph.

        Toggling a checkbox sets the `hidden` flag on matching edges via
        edges.update(), which preserves node positions (unlike removing edges).
        Renders nothing when there are no edge types to filter.
        """
        if not rel_types:
            return ""
        panel_bg, panel_border = self._panel_style()
        font_color = self.theme["font_color"]

        def _display(rel_type: str) -> str:
            return rel_type.replace('_', ' ').title()

        rows = "".join(
            '<label style="display:flex;align-items:center;gap:6px;margin:2px 0;'
            'cursor:pointer;white-space:nowrap;">'
            f'<input type="checkbox" class="nodus-edge-filter" checked '
            f'value="{json.dumps(rt)[1:-1]}" '
            'onchange="nodusApplyEdgeFilter()" style="margin:0;cursor:pointer;">'
            f'<span style="width:10px;height:10px;border-radius:50%;flex-shrink:0;'
            f'background:{self._get_color_for_relationship_type(rt)};display:inline-block;"></span>'
            f'<span>{_display(rt)}</span></label>'
            for rt in rel_types
        )

        header = (
            '<div style="display:flex;justify-content:space-between;'
            'align-items:center;gap:12px;margin-bottom:6px;font-weight:bold;">'
            '<span>Relationships</span>'
            '<span style="font-weight:normal;">'
            '<a href="#" onclick="nodusSetAllEdgeFilters(true);return false;" '
            f'style="color:{font_color};text-decoration:underline;">All</a>'
            ' / '
            '<a href="#" onclick="nodusSetAllEdgeFilters(false);return false;" '
            f'style="color:{font_color};text-decoration:underline;">None</a>'
            '</span></div>'
        )

        panel = (
            f'<div id="nodus-edge-filter" style="position:fixed;bottom:12px;left:12px;'
            f'z-index:10;background:{panel_bg};border:1px solid {panel_border};'
            f'border-radius:8px;padding:8px 12px;font-family:sans-serif;font-size:12px;'
            f'color:{font_color};max-height:40vh;overflow-y:auto;">'
            f'{header}{rows}</div>'
        )

        script = (
            '<script type="text/javascript">\n'
            'function nodusApplyEdgeFilter() {\n'
            '    if (typeof edges === "undefined") return;\n'
            '    var boxes = document.querySelectorAll(".nodus-edge-filter");\n'
            '    var visible = {};\n'
            '    boxes.forEach(function(b) { visible[b.value] = b.checked; });\n'
            '    var updates = [];\n'
            '    edges.forEach(function(e) {\n'
            '        var show = visible[e.rel_type] !== false;\n'
            '        updates.push({ id: e.id, hidden: !show });\n'
            '    });\n'
            '    if (updates.length) edges.update(updates);\n'
            '}\n'
            'function nodusSetAllEdgeFilters(checked) {\n'
            '    document.querySelectorAll(".nodus-edge-filter").forEach(function(b) {\n'
            '        b.checked = checked;\n'
            '    });\n'
            '    nodusApplyEdgeFilter();\n'
            '}\n'
            '</script>'
        )
        return panel + script

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
        # PNG export: temporarily raise devicePixelRatio (vis re-reads it in
        # setSize) so the canvas backing store densifies without its CSS size
        # ever changing — vis's setSize camera math is then a provable no-op
        # (width/height ratios are exactly 1) and the view cannot shift.
        panel_bg, panel_border = self._panel_style()
        legend_items = [[t, self._get_color_for_node_type(t)] for t in sorted(types_present)]
        export_script = (
            'var _nodusBg = ' + json.dumps(self.theme["background"]) + ';\n'
            'var _nodusFontColor = ' + json.dumps(self.theme["font_color"]) + ';\n'
            'var _nodusPanelBg = ' + json.dumps(panel_bg) + ';\n'
            'var _nodusLegendItems = ' + json.dumps(legend_items) + ';\n'
            'var _nodusExportScale = 3;\n'
            'function _nodusDrawLegend(ctx, outW, viewW) {\n'
            '    if (!_nodusLegendItems.length) return;\n'
            '    var fs = 12 * outW / viewW;\n'
            '    var lh = fs * 1.5, pad = fs, dot = fs * 0.8;\n'
            '    ctx.font = fs + "px sans-serif";\n'
            '    var maxw = 0;\n'
            '    _nodusLegendItems.forEach(function(it) {\n'
            '        maxw = Math.max(maxw, ctx.measureText(it[0]).width);\n'
            '    });\n'
            '    var bw = pad * 2 + dot + fs * 0.5 + maxw;\n'
            '    var bh = pad * 2 + lh * (_nodusLegendItems.length - 1) + fs;\n'
            '    var bx = ctx.canvas.width - bw - pad, by = pad;\n'
            '    ctx.fillStyle = _nodusPanelBg;\n'
            '    ctx.fillRect(bx, by, bw, bh);\n'
            '    ctx.textBaseline = "middle";\n'
            '    _nodusLegendItems.forEach(function(it, i) {\n'
            '        var cy = by + pad + lh * i + fs / 2;\n'
            '        ctx.fillStyle = it[1];\n'
            '        ctx.beginPath();\n'
            '        ctx.arc(bx + pad + dot / 2, cy, dot / 2, 0, 2 * Math.PI);\n'
            '        ctx.fill();\n'
            '        ctx.fillStyle = _nodusFontColor;\n'
            '        ctx.fillText(it[0], bx + pad + dot + fs * 0.5, cy);\n'
            '    });\n'
            '}\n'
            # Injected inside pyvis's drawGraph() body, so this must be
            # window-assigned to stay reachable from the button's onclick.
            # The whole cycle (fit -> render -> capture -> restore) runs
            # synchronously inside the click handler: the browser never
            # paints an intermediate frame, so the visible view never moves.
            'window.nodusExportPng = function() {\n'
            '    var btn = document.getElementById("nodus-export");\n'
            '    var container = document.getElementById("mynetwork");\n'
            '    var viewW = container.clientWidth;\n'
            '    var savedPos = network.getViewPosition();\n'
            '    var savedScale = network.getScale();\n'
            '    var dprDesc = Object.getOwnPropertyDescriptor(window, "devicePixelRatio");\n'
            '    var dpr = window.devicePixelRatio || 1;\n'
            '    btn.disabled = true;\n'
            '    try {\n'
            '        Object.defineProperty(window, "devicePixelRatio",\n'
            '            { value: dpr * _nodusExportScale, configurable: true });\n'
            '        network.setSize(' + json.dumps(self.viz_width) + ', '
                         + json.dumps(self.viz_height) + ');\n'
            '        network.fit({ animation: false });\n'
            '        network.redraw();\n'
            '        var src = network.canvas.frame.canvas;\n'
            '        var out = document.createElement("canvas");\n'
            '        out.width = src.width;\n'
            '        out.height = src.height;\n'
            '        var ctx = out.getContext("2d");\n'
            '        ctx.fillStyle = _nodusBg;\n'
            '        ctx.fillRect(0, 0, out.width, out.height);\n'
            '        ctx.drawImage(src, 0, 0);\n'
            '        _nodusDrawLegend(ctx, out.width, viewW);\n'
            '        var link = document.createElement("a");\n'
            '        link.download = "knowledge_graph.png";\n'
            '        link.href = out.toDataURL("image/png");\n'
            '        link.click();\n'
            '    } finally {\n'
            '        network.moveTo({ position: savedPos, scale: savedScale, animation: false });\n'
            '        if (dprDesc) { Object.defineProperty(window, "devicePixelRatio", dprDesc); }\n'
            '        else { delete window.devicePixelRatio; }\n'
            '        network.setSize(' + json.dumps(self.viz_width) + ', '
                         + json.dumps(self.viz_height) + ');\n'
            '        network.redraw();\n'
            '        btn.disabled = false;\n'
            '    }\n'
            '};\n'
        )
        html_content = html_content.replace(
            "network = new vis.Network(container, data, options);",
            "network = new vis.Network(container, data, options);\n" + fit_script + export_script
        )
        export_button = (
            f'<button id="nodus-export" onclick="nodusExportPng()" '
            f'style="position:fixed;top:12px;left:12px;z-index:10;'
            f'background:{panel_bg};color:{self.theme["font_color"]};'
            f'border:1px solid {panel_border};border-radius:8px;padding:6px 12px;'
            f'font-family:sans-serif;font-size:12px;cursor:pointer;">Export PNG</button>'
        )
        edge_filter = self._edge_filter_html(self._rendered_relationship_types(graph))
        html_content = html_content.replace(
            "</body>",
            self._legend_html(types_present) + export_button + edge_filter + "</body>"
        )
        logger.info("Generated HTML visualization in memory")
        return html_content

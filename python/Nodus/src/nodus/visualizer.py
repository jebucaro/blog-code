import hashlib
import logging
import webbrowser
from pathlib import Path

from pyvis.network import Network

from models import KnowledgeGraph
from settings import Settings

logger = logging.getLogger(__name__)

# Expanded color palette with good contrast
COLOR_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
]

THEMES = {
    "dark": {
        "background": "#222222",
        "font_color": "white",
        "default_node_color": "#7f7f7f",
        "edge_font_stroke": "#000000"  # Black stroke for white text
    },
    "light": {
        "background": "#ffffff",
        "font_color": "black",
        "default_node_color": "#7f7f7f",
        "edge_font_stroke": "#ffffff"  # White stroke for black text
    }
}


class GraphVisualizer:
    """Create interactive graph visualizations with PyVis"""

    def __init__(self, settings: Settings | None = None, theme: str | None = None):
        self.settings = settings or Settings()
        # Use explicit theme parameter if provided, otherwise fall back to settings
        theme_key = theme if theme is not None else self.settings.viz_theme
        self.theme = THEMES.get(theme_key, THEMES["dark"])
        self.node_type_colors = {}  # Cache for consistent colors per node type

        # Visualization settings with defaults
        self.viz_height = "100vh"
        self.viz_width = "100%"
        self.show_isolated = False
        self.physics_enabled = True

    def _get_color_for_node_type(self, node_type: str) -> str:
        """
        Get a consistent color for a node type.
        Uses hash of a node type to ensure the same type always gets the same color.
        """
        if node_type not in self.node_type_colors:
            # Use hash to get a consistent color index for this node type
            hash_value = int(hashlib.md5(node_type.encode()).hexdigest(), 16)
            color_index = hash_value % len(COLOR_PALETTE)
            self.node_type_colors[node_type] = COLOR_PALETTE[color_index]

        return self.node_type_colors[node_type]

    def _format_relationship_tooltip(self, rel_type: str, source_label: str, target_label: str) -> str:
        """Create a readable tooltip for relationships"""
        return f"{source_label} → {rel_type.replace('_', ' ').title()} → {target_label}"

    def _build_network(self, graph: KnowledgeGraph) -> Network:
        """Build a PyVis Network object from a KnowledgeGraph (internal helper)."""
        # Log node types for debugging
        node_types = set(node.type.lower() for node in graph.nodes)
        logger.info(f"Found node types in graph: {node_types}")

        # Create network
        net = Network(
            height=self.viz_height,
            width=self.viz_width,
            directed=True,
            notebook=False,
            bgcolor=self.theme["background"],
            font_color=self.theme["font_color"]
        )

        # Build lookup for valid nodes
        node_dict = {node.id: node for node in graph.nodes}

        # Filter out invalid edges and collect valid node IDs
        # Only include relationships where BOTH source and target nodes exist
        valid_edges = []
        valid_node_ids = set()
        for rel in graph.relationships:
            # Filter out self-referencing relationships (nodes targeting themselves)
            if (rel.source_node_id != rel.target_node_id and
                    rel.source_node_id in node_dict and
                    rel.target_node_id in node_dict):
                valid_edges.append(rel)
                valid_node_ids.update([rel.source_node_id, rel.target_node_id])

        # Add nodes - either all nodes or only connected ones based on config
        for node in graph.nodes:
            if self.show_isolated or node.id in valid_node_ids:
                # Use dynamic color assignment
                color = self._get_color_for_node_type(node.type.lower())

                # Enhanced tooltip with more info (with fallback for missing labels)
                display_label = node.label if node.label else ' '.join(word.capitalize() for word in node.id.split('_'))
                tooltip = f"{display_label}\nType: {node.type}\nID: {node.id}"

                try:
                    net.add_node(
                        node.id,
                        label=display_label,
                        title=tooltip,
                        color=color,
                        size=30,  # Slightly larger for better visibility
                        font={"size": 14, "color": self.theme["font_color"]}
                    )
                except Exception:
                    continue  # skip if error

        # Add valid edges with improved styling
        for rel in valid_edges:
            try:
                source_node = node_dict.get(rel.source_node_id)
                target_node = node_dict.get(rel.target_node_id)

                if source_node and target_node:
                    # Get labels with fallback for missing labels
                    source_label = source_node.label if source_node.label else ' '.join(
                        word.capitalize() for word in source_node.id.split('_'))
                    target_label = target_node.label if target_node.label else ' '.join(
                        word.capitalize() for word in target_node.id.split('_'))

                    # Create a readable tooltip for hover
                    edge_tooltip = self._format_relationship_tooltip(
                        rel.type,
                        source_label,
                        target_label
                    )

                    # Format relationship type for display
                    display_label = rel.type.replace('_', ' ').title()

                    net.add_edge(
                        rel.source_node_id,
                        rel.target_node_id,
                        label=display_label,  # Show label on the edge
                        title=edge_tooltip,  # Show detailed info on hover
                        width=2,  # Make edges more visible
                        font={"size": 12, "color": self.theme["font_color"], "strokeWidth": 2,
                              "strokeColor": self.theme["edge_font_stroke"]},
                        arrows={"to": {"enabled": True, "scaleFactor": 1.2}}
                    )
            except Exception:
                continue  # skip if error

        # Enhanced physics configuration for better layout
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
                    "stabilization": {"iterations": 100}
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

        # Log color assignments for reference
        logger.debug("Node type color assignments:")
        for node_type, color in self.node_type_colors.items():
            logger.debug(f"  {node_type}: {color}")

        return net

    def generate_html(self, graph: KnowledgeGraph) -> str:
        """
        Generate HTML visualization as a string (in-memory, no file I/O).

        Args:
            graph: KnowledgeGraph to visualize

        Returns:
            str: Complete HTML document as string
        """
        net = self._build_network(graph)

        # Use PyVis's generate_html method to get HTML string directly
        html_content = net.generate_html()

        logger.info("Generated HTML visualization in memory")
        return html_content

    def visualize(
            self,
            graph: KnowledgeGraph,
            output_file: str = "output/knowledge_graph.html",
            auto_open: bool = True
    ) -> Path:
        """
        Generate and save HTML visualization to a file.

        Args:
            graph: KnowledgeGraph to visualize
            output_file: Path where HTML file should be saved
            auto_open: Whether to open the file in browser automatically

        Returns:
            Path: Absolute path to the saved HTML file
        """
        net = self._build_network(graph)

        # Save - ensure output directory exists
        output_path = Path(output_file).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output_path))

        logger.info(f"Saved visualization to {output_path}")

        # Open in browser
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

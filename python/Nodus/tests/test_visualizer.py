from nodus.models import NODE_TYPES, KnowledgeGraph, Node, Relationship
from nodus.settings import Settings
from nodus.visualizer import NODE_TYPE_COLORS, GraphVisualizer


def make_graph():
    nodes = [
        Node(id="alice", type="person"),
        Node(id="acme", type="organization"),
        Node(id="loner", type="concept"),
    ]
    rels = [
        Relationship(id="r1", type="WORKS_AT",
                     source_node_id="alice", target_node_id="acme"),
    ]
    return KnowledgeGraph(nodes=nodes, relationships=rels)


def make_visualizer(theme="dark"):
    return GraphVisualizer(Settings(), theme=theme)


class TestColorMap:
    def test_color_map_keys_match_node_types(self):
        for theme in ("dark", "light"):
            assert set(NODE_TYPE_COLORS[theme]) == set(NODE_TYPES)

    def test_type_color_is_stable_per_theme(self):
        for theme in ("dark", "light"):
            viz = make_visualizer(theme)
            for node_type in NODE_TYPES:
                assert viz._get_color_for_node_type(node_type) == NODE_TYPE_COLORS[theme][node_type]

    def test_unmapped_type_falls_back_to_other(self):
        viz = make_visualizer("dark")
        assert viz._get_color_for_node_type("galaxy") == NODE_TYPE_COLORS["dark"]["other"]

    def test_unknown_theme_falls_back_to_dark(self):
        viz = GraphVisualizer(Settings(), theme="solarized")
        assert viz._get_color_for_node_type("person") == NODE_TYPE_COLORS["dark"]["person"]


class TestLegend:
    def test_legend_lists_present_types_only(self):
        html = make_visualizer().generate_html(make_graph())
        assert 'id="nodus-legend"' in html
        assert ">person<" in html
        assert ">organization<" in html
        # 'loner' is isolated and hidden by default, so 'concept' must not appear.
        assert ">concept<" not in html

    def test_legend_includes_isolated_types_when_shown(self):
        viz = make_visualizer()
        viz.show_isolated = True
        html = viz.generate_html(make_graph())
        assert ">concept<" in html

    def test_empty_graph_has_no_legend(self):
        html = make_visualizer().generate_html(KnowledgeGraph())
        assert 'id="nodus-legend"' not in html


class TestPlaceholderStyling:
    def test_placeholder_nodes_get_dashed_border(self):
        html = make_visualizer().generate_html(make_graph(), placeholder_ids={"acme"})
        assert "borderDashes" in html

    def test_no_placeholders_means_no_dashes(self):
        html = make_visualizer().generate_html(make_graph())
        assert "borderDashes" not in html


class TestIsolatedNodes:
    def test_isolated_hidden_by_default(self):
        html = make_visualizer().generate_html(make_graph())
        assert "loner" not in html

    def test_isolated_shown_when_enabled(self):
        viz = make_visualizer()
        viz.show_isolated = True
        html = viz.generate_html(make_graph())
        assert "loner" in html


class TestPngExport:
    def test_export_button_and_script_present(self):
        html = make_visualizer().generate_html(make_graph())
        assert 'id="nodus-export"' in html
        # Must be window-assigned: the script is injected inside pyvis's
        # drawGraph() body, so a plain function declaration would be scoped
        # to drawGraph and unreachable from the button's global onclick.
        assert "window.nodusExportPng = function" in html
        assert "function nodusExportPng()" not in html
        # Export must restore the user's exact view, not re-fit the graph.
        assert "network.getViewPosition()" in html
        assert "network.moveTo({ position: savedPos, scale: savedScale" in html
        # High-res capture must densify via devicePixelRatio, never by
        # changing the canvas CSS size (which shifts the vis camera).
        assert 'Object.defineProperty(window, "devicePixelRatio"' in html
        assert "_nodusExportScale) + \"px\"" not in html

    def test_export_legend_items_match_rendered_types(self):
        html = make_visualizer().generate_html(make_graph())
        assert '["person", "' in html
        assert '["organization", "' in html
        assert '["concept", "' not in html

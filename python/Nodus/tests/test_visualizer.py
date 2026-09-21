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


class TestNodeLabelContrast:
    """Pyvis's default node shape ('dot') draws the label outside/below the
    node, against the canvas background - not inside the node's own colored
    fill. So label color must track the theme's background/font_color pair,
    not each node's individual fill color.
    """

    def test_node_label_color_matches_theme_font_color(self):
        for theme, expected in (("dark", "white"), ("light", "black")):
            viz = make_visualizer(theme)
            html = viz.generate_html(make_graph())
            assert f'"color": "{expected}"' in html

    def test_node_label_keeps_explicit_font_size(self):
        # Regression guard: pyvis's Network(font_color=...) used to force every
        # node's font dict to {"color": ...}, silently dropping this size.
        html = make_visualizer().generate_html(make_graph())
        assert '"size": 14' in html

    def test_node_label_has_translucent_theme_matched_background(self):
        # A small background behind the label keeps it legible when an edge
        # passes underneath, without fully hiding the edge (translucent).
        for theme, expected in (
            ("dark", "rgba(34, 34, 34, 0.75)"),
            ("light", "rgba(255, 255, 255, 0.75)"),
        ):
            html = make_visualizer(theme).generate_html(make_graph())
            assert f'"background": "{expected}"' in html


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


class TestEdgeTypes:
    def _graph(self):
        nodes = [
            Node(id="alice", type="person"),
            Node(id="acme", type="organization"),
            Node(id="sf", type="location"),
        ]
        rels = [
            Relationship(id="r1", type="WORKS_AT",
                         source_node_id="alice", target_node_id="acme"),
            Relationship(id="r2", type="LIVES_IN",
                         source_node_id="alice", target_node_id="sf"),
        ]
        return KnowledgeGraph(nodes=nodes, relationships=rels)

    def test_each_edge_carries_rel_type(self):
        html = make_visualizer().generate_html(self._graph())
        assert '"rel_type": "WORKS_AT"' in html
        assert '"rel_type": "LIVES_IN"' in html

    def test_self_loops_and_dangling_edges_excluded(self):
        nodes = [Node(id="alice", type="person"), Node(id="acme", type="organization")]
        rels = [
            Relationship(id="r1", type="WORKS_AT",
                         source_node_id="alice", target_node_id="acme"),
            # self-loop: must be excluded
            Relationship(id="r2", type="KNOWS",
                         source_node_id="alice", target_node_id="alice"),
            # dangling target: must be excluded
            Relationship(id="r3", type="OWNS",
                         source_node_id="alice", target_node_id="ghost"),
        ]
        g = KnowledgeGraph(nodes=nodes, relationships=rels)
        html = make_visualizer().generate_html(g)
        assert '"rel_type": "WORKS_AT"' in html
        assert '"rel_type": "KNOWS"' not in html
        assert '"rel_type": "OWNS"' not in html


class TestEdgeColors:
    def _graph(self):
        nodes = [
            Node(id="alice", type="person"),
            Node(id="acme", type="organization"),
            Node(id="sf", type="location"),
        ]
        rels = [
            Relationship(id="r1", type="WORKS_AT",
                         source_node_id="alice", target_node_id="acme"),
            Relationship(id="r2", type="LIVES_IN",
                         source_node_id="alice", target_node_id="sf"),
        ]
        return KnowledgeGraph(nodes=nodes, relationships=rels)

    def test_color_is_stable_for_the_same_type(self):
        viz = make_visualizer()
        assert (viz._get_color_for_relationship_type("WORKS_AT")
                == viz._get_color_for_relationship_type("WORKS_AT"))

    def test_different_types_get_different_colors(self):
        viz = make_visualizer()
        assert (viz._get_color_for_relationship_type("WORKS_AT")
                != viz._get_color_for_relationship_type("LIVES_IN"))

    def test_color_is_an_hsl_string(self):
        viz = make_visualizer()
        assert viz._get_color_for_relationship_type("WORKS_AT").startswith("hsl(")

    def test_edges_carry_their_type_color_in_html(self):
        viz = make_visualizer()
        html = viz.generate_html(self._graph())
        works_at_color = viz._get_color_for_relationship_type("WORKS_AT")
        lives_in_color = viz._get_color_for_relationship_type("LIVES_IN")
        assert f'"color": "{works_at_color}"' in html
        assert f'"color": "{lives_in_color}"' in html

    def test_filter_panel_swatch_matches_edge_color(self):
        viz = make_visualizer()
        html = viz.generate_html(self._graph())
        works_at_color = viz._get_color_for_relationship_type("WORKS_AT")
        assert f"background:{works_at_color};" in html


class TestEdgeFilter:
    def _graph(self):
        nodes = [
            Node(id="alice", type="person"),
            Node(id="acme", type="organization"),
            Node(id="sf", type="location"),
        ]
        rels = [
            Relationship(id="r1", type="WORKS_AT",
                         source_node_id="alice", target_node_id="acme"),
            Relationship(id="r2", type="LIVES_IN",
                         source_node_id="alice", target_node_id="sf"),
        ]
        return KnowledgeGraph(nodes=nodes, relationships=rels)

    def test_filter_panel_present_with_checkbox_per_type(self):
        html = make_visualizer().generate_html(self._graph())
        assert 'id="nodus-edge-filter"' in html
        assert 'value="WORKS_AT"' in html
        assert 'value="LIVES_IN"' in html
        # Human-readable label rendered next to the checkbox.
        assert ">Works At<" in html
        assert ">Lives In<" in html

    def test_filter_toggles_hidden_via_edges_dataset(self):
        html = make_visualizer().generate_html(self._graph())
        assert "function nodusApplyEdgeFilter" in html
        assert "function nodusSetAllEdgeFilters" in html
        # Must toggle via the DataSet update (preserves layout), not removal.
        assert "edges.update(updates)" in html
        assert "hidden: !show" in html

    def test_all_checkboxes_checked_by_default(self):
        html = make_visualizer().generate_html(self._graph())
        # Every filter checkbox starts checked so all edges show initially.
        assert html.count('class="nodus-edge-filter" checked') == 2

    def test_no_filter_panel_for_empty_graph(self):
        html = make_visualizer().generate_html(KnowledgeGraph())
        assert 'id="nodus-edge-filter"' not in html

    def test_filter_only_lists_rendered_edge_types(self):
        # Self-loop and dangling edges must not appear as filter options.
        nodes = [Node(id="alice", type="person"), Node(id="acme", type="organization")]
        rels = [
            Relationship(id="r1", type="WORKS_AT",
                         source_node_id="alice", target_node_id="acme"),
            Relationship(id="r2", type="KNOWS",
                         source_node_id="alice", target_node_id="alice"),
            Relationship(id="r3", type="OWNS",
                         source_node_id="alice", target_node_id="ghost"),
        ]
        g = KnowledgeGraph(nodes=nodes, relationships=rels)
        html = make_visualizer().generate_html(g)
        assert 'value="WORKS_AT"' in html
        assert 'value="KNOWS"' not in html
        assert 'value="OWNS"' not in html


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

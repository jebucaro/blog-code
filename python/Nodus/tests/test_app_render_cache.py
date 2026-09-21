from unittest.mock import patch

import pytest
import streamlit as st

from nodus.app import _build_graph_html
from nodus.models import KnowledgeGraph, Node, Relationship
from nodus.visualizer import GraphVisualizer


def make_graph(context: str | None = None) -> KnowledgeGraph:
    nodes = [Node(id="alice", type="person", context=context), Node(id="acme", type="organization")]
    rels = [Relationship(id="r1", type="WORKS_AT", source_node_id="alice", target_node_id="acme")]
    return KnowledgeGraph(nodes=nodes, relationships=rels)


@pytest.fixture(autouse=True)
def clear_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def render(graph, theme="dark", show_isolated=False, placeholder_ids=()):
    return _build_graph_html(graph, graph.model_dump_json(), theme, show_isolated, frozenset(placeholder_ids))


class TestRenderGraphHtmlCache:
    def test_same_inputs_generate_the_html_only_once(self):
        with patch.object(GraphVisualizer, "generate_html", return_value="<html/>") as gen:
            render(make_graph())
            render(make_graph())
        assert gen.call_count == 1

    def test_returns_the_generated_html(self):
        with patch.object(GraphVisualizer, "generate_html", return_value="<html>x</html>"):
            assert render(make_graph()) == "<html>x</html>"

    @pytest.mark.parametrize(
        "changed",
        [
            {"theme": "light"},
            {"show_isolated": True},
            {"placeholder_ids": ("alice",)},
        ],
    )
    def test_changing_any_render_input_regenerates(self, changed):
        with patch.object(GraphVisualizer, "generate_html", return_value="<html/>") as gen:
            render(make_graph())
            render(make_graph(), **changed)
        assert gen.call_count == 2

    def test_changing_the_graph_content_regenerates(self):
        with patch.object(GraphVisualizer, "generate_html", return_value="<html/>") as gen:
            render(make_graph())
            render(make_graph(context="Joined in 2019."))
        assert gen.call_count == 2

    def test_visualizer_receives_theme_isolated_flag_and_placeholders(self):
        seen = {}

        def fake_generate(self, graph, placeholder_ids=None):
            seen.update(theme=self.theme_name, isolated=self.show_isolated, ph=placeholder_ids)
            return "<html/>"

        with patch.object(GraphVisualizer, "generate_html", fake_generate):
            render(make_graph(), theme="light", show_isolated=True, placeholder_ids=("alice",))
        assert seen == {"theme": "light", "isolated": True, "ph": {"alice"}}

    def test_graph_key_not_graph_content_drives_the_cache(self):
        with patch.object(GraphVisualizer, "generate_html", return_value="<html/>") as gen:
            _build_graph_html(make_graph(), "v1", "dark", False, frozenset())
            _build_graph_html(make_graph(context="different"), "v1", "dark", False, frozenset())
            _build_graph_html(make_graph(), "v2", "dark", False, frozenset())
        assert gen.call_count == 2

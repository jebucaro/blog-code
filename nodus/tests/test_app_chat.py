from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from nodus.models import KnowledgeGraph, Node, Relationship
from nodus.visualizer import GraphVisualizer

APP = Path(__file__).parent.parent / "src/nodus/main.py"


class FakeChatSession:
    def __init__(self):
        self.calls = []

    def ask(self, question, graph=None, source_text=None):
        self.calls.append((question, graph, source_text))
        return f"answer to {question}"


@pytest.fixture
def app():
    graph = KnowledgeGraph(
        nodes=[Node(id="a", type="person"), Node(id="b", type="organization")],
        relationships=[
            Relationship(id="r", type="WORKS_AT", source_node_id="a", target_node_id="b")
        ],
    )
    session = FakeChatSession()
    with patch("nodus.app.DocumentChatSession", lambda client, model: session):
        at = AppTest.from_file(str(APP), default_timeout=30)
        at.session_state["knowledge_graph"] = graph
        at.session_state["repair_report"] = None
        at.session_state["extracted_text"] = "the source text"
        at.session_state["extractor"] = MagicMock()
        at.run()
        yield at, session


class TestChatSection:
    def test_question_is_answered_without_an_error(self, app):
        at, _ = app
        at.chat_input[0].set_value("hello").run()
        assert not at.exception
        assert not at.error
        assert [m["content"] for m in at.session_state["chat_messages"]] == [
            "hello",
            "answer to hello",
        ]

    def test_first_question_passes_graph_and_pinned_source_text(self, app):
        at, session = app
        at.chat_input[0].set_value("hello").run()
        question, graph, source_text = session.calls[0]
        assert (question, source_text) == ("hello", "the source text")
        assert graph is not None

    def test_asking_does_not_rebuild_the_graph_visualization(self, app):
        at, _ = app
        with patch.object(GraphVisualizer, "generate_html", return_value="<html/>") as gen:
            at.chat_input[0].set_value("hello").run()
        # The render is cached by _build_graph_html; a chat message must not add
        # a fresh one.
        assert gen.call_count == 0

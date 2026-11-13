from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from nodus.app import INPUT_TYPE_OPTIONS
from nodus.models import ExtractionResult, KnowledgeGraph, Node, Relationship
from nodus.repair import RepairReport, repair_graph

APP = Path(__file__).parent.parent / "src/nodus/main.py"

TRANSCRIPT_LABEL = "Transcript (meeting, video, podcast)"
SUMMARY_LABEL = "Use executive summary to build knowledge graph"
TRANSCRIPT_SUMMARY_CAPTION = (
    "Not available in Transcript mode: the summary drops who said what, "
    "so the graph is always built from the full text."
)


def sidebar_widget(widgets, label):
    return next(w for w in widgets if w.label == label)


@pytest.fixture
def at(monkeypatch):
    monkeypatch.delenv("INPUT_TYPE", raising=False)
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    return app


@pytest.fixture
def extraction(monkeypatch):
    """App with a fake extractor and a spying repair_graph; no network access."""
    monkeypatch.delenv("INPUT_TYPE", raising=False)
    graph = KnowledgeGraph(
        nodes=[Node(id="a", type="person"), Node(id="b", type="organization")],
        relationships=[
            Relationship(id="r", type="WORKS_AT", source_node_id="a", target_node_id="b")
        ],
    )
    extractor = MagicMock()
    extractor.extract_with_summary.return_value = ExtractionResult(knowledge_graph=graph)
    repair_mock = MagicMock(wraps=repair_graph)
    with (
        patch("nodus.app.GeminiExtractor", return_value=extractor),
        patch("nodus.app.repair_graph", repair_mock),
    ):
        app = AppTest.from_file(str(APP), default_timeout=30)
        app.run()
        sidebar_widget(app.sidebar.text_input, "Gemini API key").set_value("test-key")
        app.run()
        yield app, extractor, repair_mock


def click_extract(app, text):
    app.text_area(key="input_text").set_value(text)
    next(b for b in app.button if "Extract" in b.label).click()
    app.run()
    assert not app.exception


def test_options_constant():
    assert INPUT_TYPE_OPTIONS == {"Document": "document", TRANSCRIPT_LABEL: "transcript"}


def test_selector_defaults_to_document(at):
    assert sidebar_widget(at.sidebar.selectbox, "Input type").value == "Document"
    assert at.session_state["settings"].input_type == "document"
    assert not sidebar_widget(at.sidebar.checkbox, SUMMARY_LABEL).disabled


def test_choosing_transcript_updates_settings_and_disables_summary_checkbox(at):
    sidebar_widget(at.sidebar.selectbox, "Input type").set_value(TRANSCRIPT_LABEL).run()
    assert at.session_state["settings"].input_type == "transcript"
    assert sidebar_widget(at.sidebar.checkbox, SUMMARY_LABEL).disabled


def test_summary_tick_is_hidden_in_transcript_and_restored_in_document(at):
    sidebar_widget(at.sidebar.checkbox, SUMMARY_LABEL).set_value(True).run()
    sidebar_widget(at.sidebar.selectbox, "Input type").set_value(TRANSCRIPT_LABEL).run()
    summary_box = sidebar_widget(at.sidebar.checkbox, SUMMARY_LABEL)
    assert summary_box.disabled
    assert summary_box.value is False
    assert at.session_state["use_summary_for_kg"] is True

    sidebar_widget(at.sidebar.selectbox, "Input type").set_value("Document").run()
    assert at.session_state["settings"].input_type == "document"
    summary_box = sidebar_widget(at.sidebar.checkbox, SUMMARY_LABEL)
    assert not summary_box.disabled
    assert summary_box.value is True


def test_input_type_can_be_switched_back_and_forth_without_snapping_back(at):
    for label, value in [
        ("Document", "document"),
        (TRANSCRIPT_LABEL, "transcript"),
        ("Document", "document"),
        (TRANSCRIPT_LABEL, "transcript"),
        ("Document", "document"),
    ]:
        sidebar_widget(at.sidebar.selectbox, "Input type").set_value(label).run()
        assert at.session_state["settings"].input_type == value
        assert sidebar_widget(at.sidebar.selectbox, "Input type").value == label


def test_input_type_env_var_preselects_transcript(monkeypatch):
    monkeypatch.setenv("INPUT_TYPE", "transcript")
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    assert sidebar_widget(app.sidebar.selectbox, "Input type").value == TRANSCRIPT_LABEL
    assert app.session_state["settings"].input_type == "transcript"
    sidebar_widget(app.sidebar.selectbox, "Input type").set_value("Document").run()
    assert app.session_state["settings"].input_type == "document"
    assert sidebar_widget(app.sidebar.selectbox, "Input type").value == "Document"


def sidebar_captions(app):
    return [c.value for c in app.sidebar.caption]


def test_transcript_caption_explains_summary_is_unavailable(at):
    assert TRANSCRIPT_SUMMARY_CAPTION not in sidebar_captions(at)
    sidebar_widget(at.sidebar.selectbox, "Input type").set_value(TRANSCRIPT_LABEL).run()
    assert TRANSCRIPT_SUMMARY_CAPTION in sidebar_captions(at)


def test_transcript_extraction_protects_speaker_links(extraction):
    app, extractor, repair_mock = extraction
    sidebar_widget(app.sidebar.checkbox, SUMMARY_LABEL).set_value(True).run()
    sidebar_widget(app.sidebar.selectbox, "Input type").set_value(TRANSCRIPT_LABEL).run()
    click_extract(app, "Ana: hello there")
    assert repair_mock.call_args.kwargs["protect_speaker_links"] is True
    assert extractor.extract_with_summary.call_args.kwargs["use_summary_for_kg"] is False


def test_document_extraction_does_not_protect_speaker_links(extraction):
    app, _, repair_mock = extraction
    click_extract(app, "Some article text")
    assert repair_mock.call_args.kwargs["protect_speaker_links"] is False


def test_document_extraction_follows_the_summary_checkbox(extraction):
    app, extractor, _ = extraction
    sidebar_widget(app.sidebar.checkbox, SUMMARY_LABEL).set_value(True).run()
    click_extract(app, "Some article text")
    assert extractor.extract_with_summary.call_args.kwargs["use_summary_for_kg"] is True


def test_switching_input_type_rebuilds_the_extractor(monkeypatch):
    monkeypatch.delenv("INPUT_TYPE", raising=False)
    graph = KnowledgeGraph(
        nodes=[Node(id="a", type="person"), Node(id="b", type="organization")],
        relationships=[
            Relationship(id="r", type="WORKS_AT", source_node_id="a", target_node_id="b")
        ],
    )
    with patch(
        "nodus.app.GeminiExtractor",
        side_effect=lambda settings: MagicMock(
            settings=settings.model_copy(),
            extract_with_summary=MagicMock(
                return_value=ExtractionResult(knowledge_graph=graph)
            ),
        ),
    ) as constructor:
        app = AppTest.from_file(str(APP), default_timeout=30)
        app.run()
        sidebar_widget(app.sidebar.text_input, "Gemini API key").set_value("test-key")
        app.run()
        click_extract(app, "Some article text")
        assert constructor.call_count == 1
        click_extract(app, "Some article text")
        assert constructor.call_count == 1
        sidebar_widget(app.sidebar.selectbox, "Input type").set_value(TRANSCRIPT_LABEL).run()
        click_extract(app, "Ana: hello there")
        assert constructor.call_count == 2


def test_repair_caption_reports_dropped_speaker_links():
    graph = KnowledgeGraph(
        nodes=[Node(id="a", type="person"), Node(id="b", type="organization")],
        relationships=[
            Relationship(id="r", type="WORKS_AT", source_node_id="a", target_node_id="b")
        ],
    )
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["knowledge_graph"] = graph
    app.session_state["repair_report"] = RepairReport(dropped_speaker_links=2)
    app.session_state["extracted_text"] = "text"
    app.session_state["extractor"] = MagicMock()
    app.run()
    assert not app.exception
    assert any("dropped 2 unverifiable speaker link(s)" in c.value for c in app.caption)

import logging
import uuid

import streamlit as st
from streamlit.errors import StreamlitInvalidLayoutContextError

from nodus.chat import DocumentChatSession
from nodus.errors import ExtractionError
from nodus.extractor import GeminiExtractor
from nodus.models import KnowledgeGraph, sort_relationships
from nodus.repair import repair_graph
from nodus.settings import AVAILABLE_MODELS, DEFAULT_MODEL, MAX_INPUT_LENGTH, Settings
from nodus.visualizer import GraphVisualizer

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def _build_graph_html(
    _graph: KnowledgeGraph,
    graph_key: str,
    theme: str,
    show_isolated: bool,
    placeholder_ids: frozenset[str],
) -> str:
    """Render the graph visualization once per distinct input.

    Streamlit reruns the whole script (every tab) on any interaction, so
    without this the graph HTML was rebuilt on every chat message. The graph
    is excluded from hashing (underscore prefix); `graph_key` identifies it instead.
    """
    visualizer = GraphVisualizer(theme=theme)
    visualizer.show_isolated = show_isolated
    return visualizer.generate_html(
        _graph, placeholder_ids=placeholder_ids
    )


THINKING_LEVEL_OPTIONS = {
    "Model default": "default",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
}


class StreamlitApp:
    """Streamlit application for extracting knowledge graph from text."""

    def __init__(self):
        self.setup_page()
        self.initialize_session_state()

    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Nodus",
            page_icon=":material/hub:",
            initial_sidebar_state="auto",
        )

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'knowledge_graph' not in st.session_state:
            st.session_state['knowledge_graph'] = None
        if 'graph_version' not in st.session_state:
            st.session_state['graph_version'] = None
        if 'repair_report' not in st.session_state:
            st.session_state['repair_report'] = None
        if 'executive_summary' not in st.session_state:
            st.session_state['executive_summary'] = None
        if 'settings' not in st.session_state:
            st.session_state['settings'] = Settings()
        if 'extractor' not in st.session_state:
            st.session_state['extractor'] = None
        if 'file_uploader_key' not in st.session_state:
            st.session_state['file_uploader_key'] = 0
        if 'use_summary_for_kg' not in st.session_state:
            st.session_state['use_summary_for_kg'] = False
        if 'chat_session' not in st.session_state:
            st.session_state['chat_session'] = None
        if 'chat_messages' not in st.session_state:
            st.session_state['chat_messages'] = []
        if 'extracted_text' not in st.session_state:
            st.session_state['extracted_text'] = None

    def render_sidebar(self) -> None:
        """Render sidebar with settings"""
        with st.sidebar:
            st.session_state['settings'].gemini_api_key = st.sidebar.text_input(
                "Gemini API key",
                type="password",
                placeholder="Enter Gemini API key",
                help="Enter Gemini API key to extract knowledge graph from text.",
                value=st.session_state['settings'].gemini_api_key,
            )

            current_model = st.session_state['settings'].gemini_model
            if current_model not in AVAILABLE_MODELS:
                logger.warning(
                    "Configured model '%s' is not available; falling back to '%s'.",
                    current_model, DEFAULT_MODEL,
                )
                current_model = DEFAULT_MODEL
            st.session_state['settings'].gemini_model = st.sidebar.selectbox(
                "Model",
                options=AVAILABLE_MODELS,
                index=AVAILABLE_MODELS.index(current_model),
                help="Choose the Gemini model for knowledge graph extraction.",
            )

            st.session_state['use_summary_for_kg'] = st.sidebar.checkbox(
                "Use executive summary to build knowledge graph",
                value=st.session_state['use_summary_for_kg'],
                help=(
                    "If enabled, the app first asks Gemini for an executive "
                    "summary of your text and then builds the knowledge graph "
                    "from that summary. This produces a higher-level, more "
                    "focused graph, but the summary drops most fine detail "
                    "(dates, numbers, reasons), so nodes and relationships get "
                    "little supporting context. Off by default: the graph is "
                    "built from your full text."
                ),
            )

            thinking_labels = list(THINKING_LEVEL_OPTIONS)
            current_level = st.session_state['settings'].thinking_level
            current_index = next(
                (i for i, label in enumerate(thinking_labels)
                 if THINKING_LEVEL_OPTIONS[label] == current_level),
                0,
            )
            selected_label = st.sidebar.selectbox(
                "Thinking level",
                options=thinking_labels,
                index=current_index,
                help=(
                    "Controls how much reasoning Gemini uses during knowledge graph "
                    "extraction. Higher levels are more accurate but slower. "
                    "'Model default' uses each model's built-in level."
                ),
            )
            st.session_state['settings'].thinking_level = THINKING_LEVEL_OPTIONS[selected_label]

            st.caption(
                "Nodus extracts entities and relationships from text using Gemini. "
                "Paste text or upload a file, then click Extract."
            )

    def render_main_content(self):
        """Render the main content with input text and extract button"""
        st.title(":material/hub: Nodus")

        with st.container(border=True):
            uploaded_file = st.file_uploader(
                ":material/upload_file: Upload a text file (optional)",
                type=['txt', 'md', 'text'],
                help="Upload a text file to automatically populate the text area below",
                key=f'file_uploader_{st.session_state.file_uploader_key}',
            )

            if uploaded_file is not None:
                try:
                    content = uploaded_file.read().decode('utf-8')

                    lines = content.count('\n') + 1
                    if len(content) > MAX_INPUT_LENGTH:
                        st.warning(
                            f":warning: File **{uploaded_file.name}** is {len(content):,} characters, "
                            f"which exceeds the maximum of {MAX_INPUT_LENGTH:,}. It will be truncated.")
                        st.session_state['input_text'] = content[:MAX_INPUT_LENGTH]
                    else:
                        st.toast(f"Loaded **{uploaded_file.name}** ({len(content):,} characters, {lines} lines)", icon=":material/check_circle:")
                        st.session_state['input_text'] = content
                except UnicodeDecodeError:
                    st.error(":x: Could not read file. Please ensure it's a text file in UTF-8 encoding.")
                except Exception as e:
                    st.error(f":x: Error reading file: {e}")
                    logger.exception("Error reading uploaded file")

            sample_text = st.text_area(
                "Enter text to extract knowledge graph",
                placeholder="Enter text to extract knowledge graph (or upload a file above)",
                help=f"Enter any text to extract entities and relationships from. Maximum {MAX_INPUT_LENGTH:,} characters.",
                key='input_text',
                max_chars=MAX_INPUT_LENGTH,
            )

        with st.container(horizontal=True):
            extract_button = st.button(
                ":material/auto_awesome: Extract",
                type="primary",
                width="stretch",
            )
            st.button(
                ":material/delete_sweep: Clear",
                type="secondary",
                width="stretch",
                on_click=self.reset_app_state,
            )

        if extract_button:
            if not sample_text.strip():
                st.warning(":warning: Please enter text to extract knowledge graph.")
            elif not st.session_state['settings'].gemini_api_key:
                st.warning("Enter your Gemini API key in the sidebar to get started.", icon=":material/key:")
            else:
                self.run_extraction(sample_text)

        if st.session_state['knowledge_graph']:
            self.display_results()
        elif not sample_text.strip():
            st.caption("Paste text or upload a file above, then click Extract.")

    def run_extraction(self, sample_text: str) -> None:
        """Extract executive summary and knowledge graph using Gemini API."""
        # Defense-in-depth: validate length even though text_area has max_chars,
        # because file upload path can truncate and update session_state directly
        if len(sample_text) > MAX_INPUT_LENGTH:
            st.error(f":x: Input text is too long ({len(sample_text):,} characters). Maximum allowed is {MAX_INPUT_LENGTH:,} characters.")
            logger.warning(f"Input length exceeded: {len(sample_text)} characters")
            return

        extractor = st.session_state['extractor']
        if extractor is not None:
            s = st.session_state['settings']
            if (extractor.settings.gemini_model != s.gemini_model or
                    extractor.settings.thinking_level != s.thinking_level):
                extractor.close()
                st.session_state['extractor'] = None

        # Reset chat state up front, before any extraction attempt, so a chat
        # session built on a since-closed extractor's client never survives a
        # failed re-extraction (extractor.close() above may have already run).
        st.session_state['chat_session'] = None
        st.session_state['chat_messages'] = []

        try:
            with st.status(":material/auto_awesome: Extracting knowledge graph...", expanded=True) as status:
                def progress(msg: str) -> None:
                    status.write(msg)

                if st.session_state['extractor'] is None:
                    st.session_state['extractor'] = GeminiExtractor(
                        st.session_state['settings']
                    )
                use_summary_for_kg = st.session_state.get('use_summary_for_kg', False)

                result = st.session_state['extractor'].extract_with_summary(
                    sample_text,
                    use_summary_for_kg=use_summary_for_kg,
                    show_summary=True,
                    on_progress=progress,
                )
                status.update(label=":material/check_circle: Done.", state="complete", expanded=False)

                st.session_state['executive_summary'] = result.summary
                knowledge_graph = result.knowledge_graph

                if not knowledge_graph.nodes or not knowledge_graph.relationships:
                    st.session_state['knowledge_graph'] = None
                    st.session_state['repair_report'] = None
                    st.session_state['extracted_text'] = None
                    st.warning(
                        ":warning: Extraction completed but returned an empty knowledge graph. "
                        "Try providing more detailed text or a different passage."
                    )
                    logger.info("Extraction returned an empty knowledge graph.")
                    return

                repaired_graph, repair_report = repair_graph(knowledge_graph)
                st.session_state['knowledge_graph'] = repaired_graph
                st.session_state["graph_version"] = uuid.uuid4().hex
                st.session_state['repair_report'] = repair_report
                st.session_state['extracted_text'] = sample_text

                st.toast("Knowledge graph extracted successfully.", icon=":material/check_circle:")
                logger.info("Summary and knowledge graph extracted successfully.")
        except ExtractionError as e:
            st.error(f":x: {e.user_message}")
            logger.error(f"ExtractionError while extracting knowledge graph: {e}")
        except Exception:
            st.error(":x: An unexpected error occurred while extracting the knowledge graph.")
            logger.exception("Unexpected error extracting knowledge graph")

    def display_results(self) -> None:
        """Display the extracted knowledge graph"""
        st.header("Results")
        tab_summary, tab_vis, tab_raw, tab_stats = st.tabs([
            ":material/summarize: Summary",
            ":material/hub: Graph",
            ":material/data_object: Raw data",
            ":material/analytics: Statistics",
        ])

        with tab_summary:
            self.display_summary()

        with tab_vis:
            self.display_visualization()

        with tab_raw:
            self.display_raw_data()

        with tab_stats:
            self.display_statistics()

        # Chat lives outside st.tabs() rather than as a 5th tab: st.chat_input
        # nested inside st.tabs() hits a known Streamlit bug where a rerun it
        # triggers can reset the tab selection back to whichever tab was active
        # during the last server-side rerun (streamlit/streamlit#13341). A
        # bordered container keeps the chat input "inline" (per st.chat_input's
        # own docs) instead of pinning to the bottom of the whole page, which is
        # what happens when it's called from the unwrapped main body.
        st.divider()
        st.subheader(":material/chat: Chat")
        with st.container(border=True):
            self.display_chat()

    def display_summary(self) -> None:
        """Display the executive summary, if available."""
        summary = st.session_state.get('executive_summary')

        if not summary:
            st.info(":information_source: No summary available. Run an extraction first.")
            return

        st.subheader("Executive summary")
        with st.container(border=True):
            st.write(summary.summary)

        if summary.key_points:
            st.markdown("**Key points:**")
            with st.container(border=True):
                st.markdown("\n".join(f"- {point}" for point in summary.key_points))

        summary_text = summary.summary
        if summary.key_points:
            summary_text += "\n\nKey Points:\n" + "\n".join(f"- {p}" for p in summary.key_points)

        st.download_button(
            label=":material/download: Download summary (TXT)",
            data=summary_text,
            file_name="executive_summary.txt",
            mime="text/plain",
            width="stretch",
        )

    def display_visualization(self):
        """Display the extracted knowledge graph visualization"""
        if not st.session_state.get("knowledge_graph"):
            st.info(":information_source: No knowledge graph data available to visualize.")
            return

        report = st.session_state.get('repair_report')
        has_isolated = bool(report and report.isolated_nodes)
        show_isolated = st.toggle(
            "Show isolated nodes",
            value=False,
            disabled=not has_isolated,
            help=(
                "Include nodes that have no relationships in the visualization."
                if has_isolated
                else "This graph has no isolated nodes — every node participates in a relationship."
            ),
        )
        if report and report.has_repairs:
            parts = []
            if report.remapped_endpoints:
                parts.append(f"repaired {len(report.remapped_endpoints)} edge reference(s)")
            if report.placeholder_nodes:
                parts.append(f"added {len(report.placeholder_nodes)} placeholder node(s)")
            if report.dropped_self_loops:
                parts.append(f"dropped {report.dropped_self_loops} self-loop(s)")
            if report.isolated_nodes:
                state = "shown" if show_isolated else "hidden"
                parts.append(f"{len(report.isolated_nodes)} isolated node(s) {state}")
            st.caption(":material/build: " + " · ".join(parts))

        try:
            try:
                theme = "dark" if st.context.theme.type == "dark" else "light"
            except (AttributeError, KeyError):
                theme = st.session_state['settings'].viz_theme
            placeholder_ids = frozenset(report.placeholder_nodes) if report else frozenset()
            graph = st.session_state["knowledge_graph"]
            html_content = _build_graph_html(
                graph,
                # Set once per extraction; the JSON fallback covers a graph placed in
                # session state without going through extract.
                st.session_state.get("graph_version") or graph.model_dump_json(),
                theme,
                show_isolated,
                placeholder_ids,
            )

            st.download_button(
                label=":material/download: Download visualization (HTML)",
                data=html_content,
                file_name="knowledge_graph.html",
                mime="text/html",
                width="stretch"
            )

            st.iframe(html_content, height=850)

        except Exception:
            st.error(":x: There was a problem rendering the visualization. The raw data is still available below.")
            logger.exception("Error displaying visualization")

    def display_raw_data(self):
        """Display raw knowledge graph data."""
        if st.session_state.knowledge_graph:
            import json

            kg = st.session_state.knowledge_graph

            # Presentation-only sorting: group relationships by type, then
            # source, then target (case-insensitive), and order nodes by
            # (type, id) for consistent scanning. session_state is untouched.
            sorted_rels = sort_relationships(kg.relationships)
            sorted_nodes = sorted(
                kg.nodes, key=lambda node: (node.type.lower(), node.id.lower())
            )

            # Build a sorted copy of the dump so the export/JSON view match
            # the tables, without mutating the stored graph.
            sorted_dump = kg.model_dump()
            sorted_dump["nodes"] = [node.model_dump() for node in sorted_nodes]
            sorted_dump["relationships"] = [rel.model_dump() for rel in sorted_rels]

            json_data = json.dumps(sorted_dump, indent=2)
            st.download_button(
                label=":material/download: Download data (JSON)",
                data=json_data,
                file_name="knowledge_graph.json",
                mime="application/json",
                width="stretch"
            )

            st.subheader("Nodes")
            nodes_data = [
                {
                    "ID": node.id,
                    "Label": node.label,
                    "Type": node.type,
                    "Context": node.context or "",
                }
                for node in sorted_nodes
            ]
            st.dataframe(nodes_data, width='stretch')

            st.subheader("Relationships")
            rels_data = [
                {
                    "Source": rel.source_node_id,
                    "Type": rel.type,
                    "Target": rel.target_node_id,
                    "Context": rel.context or "",
                }
                for rel in sorted_rels
            ]
            st.dataframe(rels_data, width='stretch')

            with st.expander("View full JSON", icon=":material/data_object:"):
                st.json(sorted_dump)
        else:
            st.info("No data to display")

    def display_statistics(self):
        """Display statistics about the knowledge graph."""
        if st.session_state.knowledge_graph:
            kg = st.session_state.knowledge_graph

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Nodes", len(kg.nodes))

            with col2:
                st.metric("Total Relationships", len(kg.relationships))

            with col3:
                rel_types = len({rel.type for rel in kg.relationships})
                st.metric("Relationship Types", rel_types)

            st.subheader("Node types distribution")
            node_types = {}
            for node in kg.nodes:
                node_types[node.type] = node_types.get(node.type, 0) + 1

            st.bar_chart(node_types)

            st.subheader("Relationship types distribution")
            rel_types_count = {}
            for rel in kg.relationships:
                rel_types_count[rel.type] = rel_types_count.get(rel.type, 0) + 1

            st.bar_chart(rel_types_count)

        else:
            st.info("No statistics to display")

    @st.fragment
    def display_chat(self) -> None:
        """Chat interface for asking questions about the document and its knowledge graph.

        A fragment, so a question reruns only this section instead of the whole
        page (which rebuilt the graph visualization and every other tab).
        """
        knowledge_graph = st.session_state.get('knowledge_graph')
        if not knowledge_graph:
            st.info(":information_source: Extract a knowledge graph first to chat about it.")
            return

        for message in st.session_state['chat_messages']:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        question = st.chat_input("Ask a question about the extracted graph...")
        if not question:
            return

        with st.chat_message("user"):
            st.write(question)

        try:
            session = st.session_state['chat_session']
            is_first_question = session is None
            if is_first_question:
                extractor = st.session_state['extractor']
                session = DocumentChatSession(
                    client=extractor.client,
                    model=st.session_state['settings'].gemini_model,
                )

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = (
                        session.ask(
                            question,
                            graph=knowledge_graph,
                            source_text=st.session_state['extracted_text'],
                        )
                        if is_first_question
                        else session.ask(question)
                    )
                st.write(answer)
            # Only persist the session once its first ask() has succeeded; storing it
            # earlier would leave chat_session set but _previous_interaction_id unset
            # if ask() raised, wrongly routing the next question down the follow-up
            # (no graph) path and causing a stuck ValueError loop.
            if is_first_question:
                st.session_state['chat_session'] = session
            # Append both messages together, only after ask() succeeds, so a failed
            # question never leaves an orphaned user turn with no reply underneath it.
            st.session_state['chat_messages'].append({"role": "user", "content": question})
            st.session_state['chat_messages'].append({"role": "assistant", "content": answer})
            # Without this, the exchange just rendered above (in script order,
            # after the history loop) stays visually stuck below chat_input until
            # some other interaction reruns the script - rerunning immediately
            # makes the history loop redraw everything, including this exchange,
            # in the correct order on top. Scoped to the fragment so the rest of
            # the page isn't rerun a second time. A fragment-scoped rerun is only
            # valid during a fragment rerun; on a full script run (e.g. under
            # AppTest) fall back to a full one so the exchange still redraws.
            try:
                st.rerun(scope="fragment")
            except StreamlitInvalidLayoutContextError:
                st.rerun()
        except ExtractionError as e:
            st.error(f":x: {e.user_message}")
            logger.error(f"ExtractionError during chat: {e}")
        except Exception:
            st.error(":x: An unexpected error occurred while answering the question.")
            logger.exception("Unexpected error during chat")

    def reset_app_state(self):
        """Callback function to clear all state when Clear button is clicked."""
        if st.session_state.extractor:
            try:
                st.session_state.extractor.close()
            except Exception:
                logger.warning("Error closing extractor", exc_info=True)

        st.session_state.knowledge_graph = None
        st.session_state.graph_version = None
        st.session_state.repair_report = None
        st.session_state.executive_summary = None
        st.session_state.chat_session = None
        st.session_state.chat_messages = []
        st.session_state.extracted_text = None
        st.session_state.extractor = None
        st.session_state.input_text = ''
        st.session_state.file_uploader_key += 1
        logger.info("Application state reset")

    def run(self):
        """Main run method to orchestrate the app."""
        self.render_sidebar()
        self.render_main_content()

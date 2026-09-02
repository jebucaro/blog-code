import logging

import streamlit as st

from nodus.extractor import GeminiExtractor
from nodus.repair import repair_graph
from nodus.settings import Settings, AVAILABLE_MODELS, DEFAULT_MODEL, MAX_INPUT_LENGTH
from nodus.visualizer import GraphVisualizer
from nodus.errors import ExtractionError

logger = logging.getLogger(__name__)

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
            st.session_state['use_summary_for_kg'] = True

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
                    "from that summary. This typically produces a higher-level, "
                    "more focused graph."
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
                on_click=self.clear_callback,
            )

        if extract_button:
            if not sample_text.strip():
                st.warning(":warning: Please enter text to extract knowledge graph.")
            elif not st.session_state['settings'].gemini_api_key:
                st.warning("Enter your Gemini API key in the sidebar to get started.", icon=":material/key:")
            else:
                self.extract_knowledge_graph(sample_text)

        if st.session_state['knowledge_graph']:
            self.display_results()
        elif not sample_text.strip():
            st.caption("Paste text or upload a file above, then click Extract.")

    def extract_knowledge_graph(self, sample_text: str) -> None:
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

        try:
            with st.status(":material/auto_awesome: Extracting knowledge graph...", expanded=True) as status:
                def progress(msg: str) -> None:
                    status.write(msg)

                if st.session_state['extractor'] is None:
                    st.session_state['extractor'] = GeminiExtractor(
                        st.session_state['settings']
                    )
                use_summary_for_kg = st.session_state.get('use_summary_for_kg', True)

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
                    st.warning(
                        ":warning: Extraction completed but returned an empty knowledge graph. "
                        "Try providing more detailed text or a different passage."
                    )
                    logger.info("Extraction returned an empty knowledge graph.")
                    return

                repaired_graph, repair_report = repair_graph(knowledge_graph)
                st.session_state['knowledge_graph'] = repaired_graph
                st.session_state['repair_report'] = repair_report

                st.toast("Knowledge graph extracted successfully.", icon=":material/check_circle:")
                logger.info("Summary and knowledge graph extracted successfully.")
        except ExtractionError as e:
            st.error(f":x: {e.user_message}")
            logger.error(f"ExtractionError while extracting knowledge graph: {e}")
        except Exception as e:
            st.error(":x: An unexpected error occurred while extracting the knowledge graph.")
            logger.exception(f"Unexpected error extracting knowledge graph: {e}")

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
                theme = "dark" if st.context.theme.base == "dark" else "light"
            except (AttributeError, KeyError):
                theme = st.session_state['settings'].viz_theme
            visualizer = GraphVisualizer(st.session_state['settings'], theme=theme)
            visualizer.show_isolated = show_isolated

            placeholder_ids = set(report.placeholder_nodes) if report else None
            html_content = visualizer.generate_html(
                st.session_state["knowledge_graph"], placeholder_ids=placeholder_ids
            )

            st.download_button(
                label=":material/download: Download visualization (HTML)",
                data=html_content,
                file_name="knowledge_graph.html",
                mime="text/html",
                width="stretch"
            )

            st.iframe(html_content, height=850)

        except Exception as e:
            st.error(":x: There was a problem rendering the visualization. The raw data is still available below.")
            logger.exception(f"Error displaying visualization: {e}")

    def display_raw_data(self):
        """Display raw knowledge graph data."""
        if st.session_state.knowledge_graph:
            import json

            json_data = json.dumps(
                st.session_state.knowledge_graph.model_dump(),
                indent=2
            )
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
                    "Type": node.type
                }
                for node in st.session_state.knowledge_graph.nodes
            ]
            st.dataframe(nodes_data, width='stretch')

            st.subheader("Relationships")
            rels_data = [
                {
                    "Source": rel.source_node_id,
                    "Type": rel.type,
                    "Target": rel.target_node_id
                }
                for rel in st.session_state.knowledge_graph.relationships
            ]
            st.dataframe(rels_data, width='stretch')

            with st.expander("View full JSON", icon=":material/data_object:"):
                st.json(st.session_state.knowledge_graph.model_dump())
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
                rel_types = len(set(rel.type for rel in kg.relationships))
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

    def clear_callback(self):
        """Callback function to clear all state when Clear button is clicked."""
        if st.session_state.extractor:
            try:
                st.session_state.extractor.close()
            except Exception as e:
                logger.warning(f"Error closing extractor: {e}")

        st.session_state.knowledge_graph = None
        st.session_state.repair_report = None
        st.session_state.executive_summary = None
        st.session_state.extractor = None
        st.session_state.input_text = ''
        st.session_state.file_uploader_key += 1
        logger.info("Application state reset")

    def run(self):
        """Main run method to orchestrate the app."""
        self.render_sidebar()
        self.render_main_content()

import logging

import streamlit as st
from streamlit.components.v1 import html

from nodus.extractor import GeminiExtractor
from nodus.settings import Settings, AVAILABLE_MODELS, MAX_INPUT_LENGTH
from nodus.visualizer import GraphVisualizer
from nodus.errors import ExtractionError

logger = logging.getLogger(__name__)


class StreamlitApp:
    """Streamlit application for extracting knowledge graph from text."""

    def __init__(self):
        self.setup_page()
        self.initialize_session_state()

    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Nodus",
            page_icon=":spider_web:",
            layout="wide",
            initial_sidebar_state="auto",
        )

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'knowledge_graph' not in st.session_state:
            st.session_state['knowledge_graph'] = None
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
            st.title("Settings")
            st.subheader("Gemini API Key")
            st.session_state['settings'].gemini_api_key = st.sidebar.text_input(
                "Enter Gemini API key",
                type="password",
                placeholder="Enter Gemini API key",
                help="Enter Gemini API key to extract knowledge graph from text.",
                value=st.session_state['settings'].gemini_api_key,
            )

            st.subheader("Model Selection")
            st.session_state['settings'].gemini_model = st.sidebar.selectbox(
                "Select Gemini Model",
                options=AVAILABLE_MODELS,
                index=AVAILABLE_MODELS.index(st.session_state['settings'].gemini_model),
                help="Choose the Gemini model for knowledge graph extraction.",
            )

            st.subheader("Pipeline Options")
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

            st.divider()

            st.markdown("""
            ## :information_source: **About Nodus**
            
            Nodus is a simple knowledge graph extraction tool powered by Google Gemini API.
            
            **Features:**
            - Extract entities and relationships
            - Interactive graph visualization
            - Export results
            
            **How to use:**
            1. Enter Gemini API key
            2. Paste text to extract knowledge graph
            3. Click "Extract" button
            4. View extracted knowledge graph
            """)

    def render_main_content(self):
        """Render the main content with input text and extract button"""
        st.title(":spider_web: Nodus")
        st.subheader("Knowledge Graph Extractor")

        uploaded_file = st.file_uploader(
            ":page_facing_up: Upload a text file (optional)",
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
                    st.success(
                        f":white_check_mark: Loaded **{uploaded_file.name}** ({len(content):,} characters, {lines} lines)")
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

        col1, col2 = st.columns([3, 1])

        with col1:
            extract_button = st.button(
                ":rocket: Extract",
                type="primary",
                use_container_width=True,
            )

        with col2:
            clear_button = st.button(
                ":broom: Clear",
                type="secondary",
                use_container_width=True,
                on_click=self.clear_callback,
            )

        if extract_button:
            if not sample_text.strip():
                st.warning(":warning: Please enter text to extract knowledge graph.")
            else:
                self.extract_knowledge_graph(sample_text)

        if st.session_state['knowledge_graph']:
            st.divider()
            self.display_results()
        elif not sample_text.strip():
            st.info(":information_source: Please enter text to extract knowledge graph.")

    def extract_knowledge_graph(self, sample_text: str) -> None:
        """Extract executive summary and knowledge graph using Gemini API."""
        # Defense-in-depth: validate length even though text_area has max_chars,
        # because file upload path can truncate and update session_state directly
        if len(sample_text) > MAX_INPUT_LENGTH:
            st.error(f":x: Input text is too long ({len(sample_text):,} characters). Maximum allowed is {MAX_INPUT_LENGTH:,} characters.")
            logger.warning(f"Input length exceeded: {len(sample_text)} characters")
            return

        try:
            with st.spinner(":mag: Summarizing text and extracting knowledge graph..."):
                if st.session_state['extractor'] is None:
                    st.session_state['extractor'] = GeminiExtractor(
                        st.session_state['settings']
                    )
                use_summary_for_kg = st.session_state.get('use_summary_for_kg', True)

                result = st.session_state['extractor'].extract_with_summary(
                    sample_text,
                    use_summary_for_kg=use_summary_for_kg,
                )

                st.session_state['executive_summary'] = result.summary
                knowledge_graph = result.knowledge_graph

                if not knowledge_graph.nodes or not knowledge_graph.relationships:
                    st.session_state['knowledge_graph'] = None
                    st.warning(
                        ":warning: Extraction completed but returned an empty knowledge graph. "
                        "Try providing more detailed text or a different passage."
                    )
                    logger.info("Extraction returned an empty knowledge graph.")
                    return

                st.session_state['knowledge_graph'] = knowledge_graph

                st.success(":white_check_mark: Summary generated and knowledge graph extracted successfully.")
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
        tab_summary, tab_vis, tab_raw, tab_stats = st.tabs(
            [":page_facing_up: Summary", ":bar_chart: Visualization", ":memo: Raw Data", ":chart_with_upwards_trend: Statistics"]
        )

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

        st.subheader("Executive Summary")
        st.write(summary.summary)

        if summary.key_points:
            st.markdown("**Key Points:**")
            for point in summary.key_points:
                st.markdown(f"- {point}")

        summary_text = summary.summary
        if summary.key_points:
            summary_text += "\n\nKey Points:\n" + "\n".join(f"- {p}" for p in summary.key_points)

        st.download_button(
            label="📥 Download Summary (TXT)",
            data=summary_text,
            file_name="executive_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

    def display_visualization(self):
        """Display the extracted knowledge graph visualization"""
        if not st.session_state.get("knowledge_graph"):
            st.info(":information_source: No knowledge graph data available to visualize.")
            return

        try:
            visualizer = GraphVisualizer(
                st.session_state['settings']
            )

            html_content = visualizer.generate_html(st.session_state["knowledge_graph"])

            st.download_button(
                label="📥 Download Visualization (HTML)",
                data=html_content,
                file_name="knowledge_graph.html",
                mime="text/html",
                use_container_width=True
            )

            html(html_content, height=768, scrolling=True)

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
                label="📥 Download Data (JSON)",
                data=json_data,
                file_name="knowledge_graph.json",
                mime="application/json",
                use_container_width=True
            )

            st.divider()

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

            st.divider()

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

            st.divider()

            with st.expander("View Full JSON"):
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

            st.divider()

            st.subheader("Node Types Distribution")
            node_types = {}
            for node in kg.nodes:
                node_types[node.type] = node_types.get(node.type, 0) + 1

            st.bar_chart(node_types)

            st.divider()

            st.subheader("Relationship Types Distribution")
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
        st.session_state.executive_summary = None
        st.session_state.extractor = None
        st.session_state.input_text = ''
        st.session_state.file_uploader_key += 1
        logger.info("Application state reset")

    def run(self):
        """Main run method to orchestrate the app."""
        self.render_sidebar()
        self.render_main_content()

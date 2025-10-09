import logging

import streamlit as st
from streamlit.components.v1 import html

from extractor import GeminiExtractor
from settings import Settings, AVAILABLE_MODELS
from visualizer import GraphVisualizer

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
            layout="centered",
            initial_sidebar_state="auto",
        )

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'knowledge_graph' not in st.session_state:
            st.session_state['knowledge_graph'] = None
        if 'settings' not in st.session_state:
            st.session_state['settings'] = Settings()
        if 'extractor' not in st.session_state:
            st.session_state['extractor'] = None
        if 'file_uploader_key' not in st.session_state:
            st.session_state['file_uploader_key'] = 0

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

        # File upload option
        uploaded_file = st.file_uploader(
            ":page_facing_up: Upload a text file (optional)",
            type=['txt', 'md', 'text'],
            help="Upload a text file to automatically populate the text area below",
            key=f'file_uploader_{st.session_state.file_uploader_key}',
        )

        if uploaded_file is not None:
            try:
                # Read file content
                content = uploaded_file.read().decode('utf-8')
                st.session_state['input_text'] = content

                # Show file info
                lines = content.count('\n') + 1
                st.success(
                    f":white_check_mark: Loaded **{uploaded_file.name}** ({len(content)} characters, {lines} lines)")
            except UnicodeDecodeError:
                st.error(":x: Could not read file. Please ensure it's a text file in UTF-8 encoding.")
            except Exception as e:
                st.error(f":x: Error reading file: {e}")

        sample_text = st.text_area(
            "Enter text to extract knowledge graph",
            placeholder="Enter text to extract knowledge graph (or upload a file above)",
            help="Enter any text to extract entities and relationships from.",
            key='input_text',
        )

        # Button layout: Extract and Clear side by side
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
                disabled=st.session_state['knowledge_graph'] is None,
                use_container_width=True,
                on_click=self.clear_callback,
            )

        if extract_button:
            if not sample_text.strip():
                st.warning(":warning: Please enter text to extract knowledge graph.")
            else:
                self.extract_knowledge_graph(sample_text)
                st.rerun()

        if st.session_state['knowledge_graph']:
            st.divider()
            self.display_results()
        elif not sample_text.strip():
            st.info(":information_source: Please enter text to extract knowledge graph.")

    def extract_knowledge_graph(self, sample_text: str) -> None:
        """Extract knowledge graph from text using Gemini API"""

        try:
            with st.spinner(":mag: Extracting knowledge graph..."):
                if st.session_state['extractor'] is None:
                    st.session_state['extractor'] = GeminiExtractor(
                        st.session_state['settings']
                    )

                knowledge_graph = st.session_state['extractor'].extract(sample_text)
                st.session_state['knowledge_graph'] = knowledge_graph

                st.success(":white_check_mark: Knowledge graph extracted successfully.")
                logger.info("Knowledge graph extracted successfully.")
        except Exception as e:
            st.error(f":x: Error extracting knowledge graph: {e}")
            logger.error(f"Error extracting knowledge graph: {e}")

    def display_results(self) -> None:
        """Display the extracted knowledge graph"""
        st.header("Results")

        tab1, tab2, tab3 = st.tabs(
            [":bar_chart: Visualization", ":memo: Raw Data", ":chart_with_upwards_trend: Statistics"])

        with tab1:
            self.display_visualization()

        with tab2:
            self.display_raw_data()

        with tab3:
            self.display_statistics()

    def display_visualization(self):
        """Display the extracted knowledge graph visualization"""
        try:
            visualizer = GraphVisualizer(
                st.session_state['settings']
            )

            # Generate HTML in memory (no file I/O)
            html_content = visualizer.generate_html(st.session_state["knowledge_graph"])

            # Add a download button for HTML
            st.download_button(
                label="📥 Download Visualization (HTML)",
                data=html_content,
                file_name="knowledge_graph.html",
                mime="text/html",
                use_container_width=True
            )

            # Display the visualization in the browser
            html(html_content, height=768, scrolling=True)

        except Exception as e:
            st.error(f":x: Error displaying visualization: {e}")
            logger.error(f"Error displaying visualization: {e}")

    def display_raw_data(self):
        """Display raw knowledge graph data."""
        if st.session_state.knowledge_graph:
            import json

            # Add a download button for JSON
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

            # Display nodes
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

            # Display relationships
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

            # Display full JSON
            with st.expander("View Full JSON"):
                st.json(st.session_state.knowledge_graph.model_dump())
        else:
            st.info("No data to display")

    def display_statistics(self):
        """Display statistics about the knowledge graph."""
        if st.session_state.knowledge_graph:
            kg = st.session_state.knowledge_graph

            # Basic metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Nodes", len(kg.nodes))

            with col2:
                st.metric("Total Relationships", len(kg.relationships))

            with col3:
                # Calculate unique relationship types
                rel_types = len(set(rel.type for rel in kg.relationships))
                st.metric("Relationship Types", rel_types)

            st.divider()

            # Node types breakdown
            st.subheader("Node Types Distribution")
            node_types = {}
            for node in kg.nodes:
                node_types[node.type] = node_types.get(node.type, 0) + 1

            st.bar_chart(node_types)

            st.divider()

            # Relationship types breakdown
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
        st.session_state.extractor = None
        st.session_state.input_text = ''
        st.session_state.file_uploader_key += 1
        logger.info("Application state reset")

    def run(self):
        """Main run method to orchestrate the app."""
        self.render_sidebar()
        self.render_main_content()

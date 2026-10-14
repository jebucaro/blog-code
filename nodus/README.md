# Nodus

A knowledge graph extraction and visualization system powered by Google Gemini AI. Nodus extracts structured entities and relationships from unstructured text and creates interactive, explorable graph visualizations.

## Features

- **AI-Powered Extraction**: Uses Google Gemini API to intelligently extract nodes (entities) and relationships from text
- **Interactive Visualizations**: Creates dynamic, explorable graph visualizations using PyVis
- **Multiple Model Support**: Choose from curated Gemini models (Flash Lite, Flash, Pro) based on your needs
- **File Upload**: Load text files directly or paste text manually
- **Export Capabilities**: Download visualizations as HTML or raw data as JSON
- **Real-time Statistics**: View node/relationship counts and type distributions
- **Semantic Deduplication**: Automatically removes duplicate relationships for clean graphs
- **Streamlit Web Interface**: User-friendly web application with tabbed results view

## Prerequisites

- **Python**: 3.12 or higher
- **UV Package Manager**: [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- **Google Gemini API Key**: [Get your API key](https://ai.google.dev/)

## Installation

### 1. Install Dependencies with UV

UV is a fast Python package manager that handles virtual environments automatically.

```bash
# Install all dependencies (UV will create a virtual environment automatically)
uv sync
```

This command:
- Creates a virtual environment in `.venv/`
- Installs all dependencies from `pyproject.toml`
- Locks dependency versions in `uv.lock`

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (defaults shown)
GEMINI_MODEL=gemini-3.8-flash
THINKING_LEVEL=default
VIZ_THEME=dark
```

Alternatively, you can enter the API key directly in the Streamlit sidebar when running the app.

## Usage

### Running the Web Application

The primary interface is a Streamlit web application:

```bash
uv run streamlit run src/nodus/main.py
```

This will:
1. Start the Streamlit server
2. Open your default browser to `http://localhost:8501`
3. Display the Nodus web interface

### Using the Web Interface

1. **Enter API Key**: Paste your Gemini API key in the sidebar (if not in `.env`)
2. **Select Model**: Choose from:
   - `gemini-3.5-flash-lite` - Fastest, most cost-effective
   - `gemini-3.8-flash` - Balanced performance (default)
   - `gemini-4-argon` - Most capable, deepest reasoning (paid tier only)
3. **Pick a Thinking Level** (optional): "Model default" uses each model's built-in
   thinking level; Low/Medium/High trade speed for extraction quality.
   The sidebar's "Use executive summary to build knowledge graph" checkbox is **off**
   by default: the graph is built from your full text, which keeps the dates, numbers,
   and reasons that fill each node's and relationship's `context`. Turn it on for a
   higher-level graph built from the summary (most fine detail is lost that way).
4. **Input Text**: Either:
   - Upload a text file (`.txt`, `.md`) using the file uploader
   - Paste text directly into the text area
5. **Extract**: Click "Extract" to generate the knowledge graph
6. **View Results**: Explore four tabs:
   - **Summary**: Document summary
   - **Graph**: Interactive graph (drag, zoom, hover for details). Use the
     relationship-type filter (bottom-left) to show or hide edges by type, with
     "All"/"None" shortcuts. Filtering hides edges without disturbing the layout.
   - **Raw data**: Tables of nodes and relationships (grouped and sorted by
     relationship type, then source, then target), plus full JSON in the same order
   - **Statistics**: Counts and distributions

   Below the tabs, the **Chat** section lets you ask questions about the document;
   answers come from the original source text. Each answer ends with an **Evidence**
   section: verbatim quotes from the document, and any fact taken from the
   relationship list listed separately, so you can see where each claim came from. The quotes are
   written by the model and can occasionally be inexact, so check anything important
   against the source.
7. **Export**: Download HTML visualization or JSON data
8. **Clear**: Reset everything and start over

### Example Text

Try this sample text:

```
Alice Johnson is a 34-year-old software engineer living in San Francisco.
She works at TechCorp as a senior developer and specializes in machine learning.
Alice enjoys hiking in her free time and often visits Yosemite National Park on weekends.
She graduated from Stanford University with a degree in Computer Science in 2012.
```

## Project Structure

```
Nodus/
├── src/nodus/
│   ├── models.py          # Pydantic models (Node, Relationship, KnowledgeGraph)
│   ├── extractor.py       # Gemini API integration for extraction
│   ├── visualizer.py      # PyVis-based graph visualization
│   ├── app.py             # Streamlit web application
│   ├── settings.py        # Configuration and settings management
│   └── main.py            # Application entry point
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependency versions
├── .env.example           # Environment variables (rename this to .env)
└── README.md              # This file
```

## Core Architecture

### 1. Extraction Layer (`extractor.py`)
- Uses Google Gemini API with structured JSON output
- Enforces strict formatting rules (lowercase IDs, uppercase relationship types)
- Handles coreference resolution for entity consistency
- Returns validated `KnowledgeGraph` objects

### 2. Data Models (`models.py`)
- **Node**: Entity with `id`, `label`, `type`, optional `context`
- **Relationship**: Connection with `id`, `type`, `source_node_id`, `target_node_id`, optional `context`
- **KnowledgeGraph**: Container with nodes and relationships
- Auto-generates labels and deduplicates relationships
- `context` holds a short (≤200 char) supporting detail - a date, number, quote fragment, or
  key nuance from the source text - when the LLM finds one worth keeping beyond the bare
  label/type/relationship fact. It's optional and often absent; shown in graph tooltips, the
  Raw Data tables. It is not sent to Chat, which gets only the relationship list.

### 3. Visualization Layer (`visualizer.py`)
- Creates interactive HTML visualizations using PyVis
- Consistent color-coding by node type (MD5 hash-based)
- Configurable themes (dark/light), physics simulation
- Filters self-referencing and invalid relationships
- Relationship-type filter overlay to show/hide edges by type without moving nodes

### 4. Web Interface (`app.py`)
- Streamlit-based UI with session state management
- File upload support for text files
- Four-tab results view (Summary, Visualization, Raw Data, Statistics), plus a Chat
  section below the tabs
- Chat: ask natural-language questions answered from the original source text, with
  a compact list of the extracted relationships as a consistency aid for entity names
- Export functionality (HTML, JSON)

## Configuration

### Available Models

The application supports three curated Gemini models:

| Model | Best For | Speed | Cost |
|-------|----------|-------|------|
| `gemini-3.5-flash-lite` | Quick extractions, small texts | Fastest | Lowest |
| `gemini-3.8-flash` | Balanced performance (default) | Fast | Medium |
| `gemini-4-argon` | Complex texts, high accuracy (paid tier only) | Slower | Higher |

### Visualization Themes

Set in `.env` or modify in code:
- `light` - White background
- `dark` - Dark background (default)

### Content Safety

Nodus calls Gemini via the Interactions API, which does not currently support
the adjustable `safety_settings` (harassment, hate speech, sexually explicit,
dangerous content). Only Gemini's always-on core protections (e.g. child
safety) apply; the adjustable filters default to **Off**. Keep this in mind if
you're feeding it untrusted or user-submitted text.

### Chat Data Retention

Knowledge graph extraction calls use `store=False` (see `extractor.py`) so
Gemini doesn't retain your document text or extracted graph server-side.
Chat is a deliberate exception: it uses `store=True` so Gemini can
keep the conversation in a server-side thread across follow-up questions
instead of resending context every turn in the request (follow-up turns still
replay the stored conversation server-side, so the document counts as input
tokens on every turn even though it isn't re-sent). This means chat conversations are
retained by Gemini for the standard Interactions API retention window (55
days on the paid tier, 1 day on the free tier).

Chat sends the **full original source text** (not just the extracted
graph) on the first question of each session, so Gemini retains your raw
document content for that retention window while you're using Chat - a bigger tradeoff than earlier versions of this app made, when chat
sent only the structured graph. Extraction and chat are otherwise
independent - the retention tradeoff only applies while you're actively
using Chat, and starts fresh (a new server-side thread) every time
you re-extract or click Clear.

## Development

### Project Dependencies

- **google-genai**: Google Gemini API client
- **pydantic**: Data validation and settings management
- **pydantic-settings**: Environment variable configuration
- **pyvis**: Interactive network graph visualization
- **streamlit**: Web application framework

## Docker Deployment

Nodus includes a production-ready Dockerfile using UV and Python 3.12-slim:

```bash
# 1. Ensure Docker Desktop is running, then build the image
docker build -t nodus:latest .

# 2. Run with environment variable
docker run -p 8501:8501 -e GEMINI_API_KEY=your_key_here nodus:latest

# 3. Or use .env file
docker run -p 8501:8501 --env-file .env nodus:latest

# 4. Access the app at http://localhost:8501
```

**Note**: The `.env` file is excluded from the Docker image for security. Environment variables must be provided at runtime using `-e` or `--env-file`. No rebuild is needed when changing environment variables.

## Troubleshooting

### API Key Issues
- Ensure your Gemini API key is valid and has a quota remaining
- Check that the key is correctly set in `.env` or manually entered in the sidebar

### Model Not Available
- Verify the model name is spelled correctly
- Some models may have regional restrictions

### File Upload Errors
- Ensure files are UTF-8 encoded text files
- Try converting files to plain text format

### Visualization Not Loading
- Check the browser console for JavaScript errors
- Try disabling browser extensions that block scripts
- Ensure sufficient RAM for large graphs (100+ nodes)

## License

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>

## Author

Jonathan Búcaro

## Acknowledgments

- Powered by [Google Gemini API](https://ai.google.dev/)
- Built with [Streamlit](https://streamlit.io/)
- Visualizations by [PyVis](https://pyvis.readthedocs.io/)
- Package management by [UV](https://docs.astral.sh/uv/)

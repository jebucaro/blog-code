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
GEMINI_MODEL=gemini-2.5-flash-lite
VIZ_THEME=light
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
   - `gemini-2.5-flash-lite` - Fastest, most cost-effective
   - `gemini-2.5-flash` - Balanced performance
   - `gemini-2.5-pro` - Most capable, larger context
3. **Input Text**: Either:
   - Upload a text file (`.txt`, `.md`) using the file uploader
   - Paste text directly into the text area
4. **Extract**: Click "Extract" to generate the knowledge graph
5. **View Results**: Explore three tabs:
   - **Visualization**: Interactive graph (drag, zoom, hover for details)
   - **Raw Data**: Tables of nodes and relationships, full JSON
   - **Statistics**: Counts and distributions
6. **Export**: Download HTML visualization or JSON data
7. **Clear**: Reset everything and start over

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
- **Node**: Entity with `id`, `label`, `type`
- **Relationship**: Connection with `id`, `type`, `source_node_id`, `target_node_id`
- **KnowledgeGraph**: Container with nodes and relationships
- Auto-generates labels and deduplicates relationships

### 3. Visualization Layer (`visualizer.py`)
- Creates interactive HTML visualizations using PyVis
- Consistent color-coding by node type (MD5 hash-based)
- Configurable themes (dark/light), physics simulation
- Filters self-referencing and invalid relationships

### 4. Web Interface (`app.py`)
- Streamlit-based UI with session state management
- File upload support for text files
- Three-tab results view (Visualization, Raw Data, Statistics)
- Export functionality (HTML, JSON)

## Configuration

### Available Models

The application supports three curated Gemini models:

| Model | Best For | Speed | Cost |
|-------|----------|-------|------|
| `gemini-2.5-flash-lite` | Quick extractions, small texts | Fastest | Lowest |
| `gemini-2.5-flash` | Balanced performance | Fast | Medium |
| `gemini-2.5-pro` | Complex texts, high accuracy | Slower | Higher |

### Visualization Themes

Set in `.env` or modify in code:
- `light` - White background (default)
- `dark` - Dark background

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

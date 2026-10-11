# Multi-stage Dockerfile for Nodus Streamlit App with UV
# Base image: Python 3.12-slim

# ==================== Builder Stage ====================
FROM python:3.12-slim AS builder

# Copy UV binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies only (not the project itself)
# Use cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy the entire project
COPY . .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ==================== Runtime Stage ====================
FROM python:3.12-slim

# Install curl for health checks
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Copy application code
COPY --chown=app:app src ./src
COPY --chown=app:app pyproject.toml ./

# Switch to non-root user
USER app

# Add .venv/bin to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
ENTRYPOINT ["streamlit", "run", "src/nodus/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

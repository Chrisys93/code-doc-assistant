FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for tree-sitter
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install --upgrade chromadb

# Copy application code
COPY src/ ./src/

# Ensure src is importable as a package
ENV PYTHONPATH=/app

# Streamlit config — disable telemetry, set port
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.headless=true"]

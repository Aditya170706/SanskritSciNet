FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        fastapi uvicorn[standard] streamlit \
        pydantic pyyaml psutil sentence-transformers sympy

# Copy project
COPY . .

# Create required dirs
RUN mkdir -p logs metrics data/raw data/cleaned data/segmented \
             data/annotated data/training database/vector_index \
             database/knowledge_graph database/search_index

# Expose ports
EXPOSE 8000 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default: start both API + UI
CMD ["python", "scripts/run_system.py"]

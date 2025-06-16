# LLMESH Network Docker Image

FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY mesh.config.yaml .
COPY setup.py .

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p /app/data /app/logs /app/models

# Expose ports
EXPOSE 8080 8888 9090

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MESH_CONFIG=/app/mesh.config.yaml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/status || exit 1

# Run node
CMD ["python", "-m", "mesh.cli", "start"]

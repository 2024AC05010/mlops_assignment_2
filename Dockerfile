# Multi-stage build for optimized image size
# Stage 1: Builder - installs dependencies and builds the application
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime - minimal image for running the application
FROM python:3.12-slim

WORKDIR /app

# Copy only the installed packages from builder stage
COPY --from=builder /root/.local /root/.local
# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/
COPY data/sample/ ./data/sample/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

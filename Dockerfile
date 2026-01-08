# Sears Home Services Technician Analytics Platform
# Python 3.11 FastAPI Application

FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install .

# Copy application code
COPY app/ ./app/

# Copy alembic for migrations
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

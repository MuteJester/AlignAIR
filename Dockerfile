FROM python:3.11-slim

# Metadata (align with package version)
LABEL version="2.0.2"
LABEL description="AlignAIR v2.0 - Unified Multi-Chain IG/TCR Sequence Alignment Tool"
LABEL maintainer="Thomas Konstantinovsky & Ayelet Peres"

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build tools and dependencies via modern PEP 517 workflow
# Copy minimal files first to maximize Docker layer caching
COPY pyproject.toml setup.py README.md MANIFEST.in requirements.txt ./
COPY src/ ./src/

RUN pip install --upgrade pip setuptools wheel \
	&& pip install .

# Copy rest of the repo (scripts, configs, etc.)
COPY . .

# Place pretrained checkpoints in a predictable path
COPY checkpoints /app/pretrained_models

EXPOSE 8000

# Create non-root user for safer defaults
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Align version env with package
ENV ALIGNAIR_VERSION=2.0.2

# Healthcheck (lightweight)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD python -c "import tensorflow as tf; print('ok')" || exit 1

# Entry point to run Typer app directly (e.g., `docker run ... run --model-dir ...`)
ENTRYPOINT ["python", "app.py"]
CMD ["doctor"]

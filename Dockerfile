# AlignAIR — neural IG/TCR sequence aligner. CPU image (the default; small + portable).
# For GPU inference/training, install AlignAIR in a CUDA base image instead (see docs).
FROM python:3.11-slim

LABEL org.opencontainers.image.title="AlignAIR" \
      org.opencontainers.image.description="Neural IG/TCR sequence aligner (AIRR rearrangement output)" \
      org.opencontainers.image.source="https://github.com/MuteJester/AlignAIR" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install a CPU-only PyTorch first (its own layer, so source edits don't re-download it),
# so the project install does not pull the large CUDA wheel.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then copy install inputs and install AlignAIR with the CLI extra
# (the parasail reader is optional: use ".[cli,reader]").
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install ".[cli]"

# Bundled example data (see examples/README.md).
COPY examples/ ./examples/

# Non-root by default.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Healthcheck = environment self-check (exits non-zero if a core dependency is missing).
HEALTHCHECK --interval=30s --timeout=20s --retries=3 CMD alignair doctor || exit 1

ENTRYPOINT ["alignair"]
CMD ["--help"]

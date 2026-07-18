# AlignAIR — neural IG/TCR sequence aligner. CPU image (the default; portable, no CUDA).
# For GPU inference/training, install AlignAIR in a CUDA base image instead (see docs).
#
# Two stages: the builder compiles the optional Cython germline-CIGAR kernel into a wheel (it needs a
# C compiler), and the runtime installs that wheel into a slim image — so the shipped image gets the
# fast kernel WITHOUT carrying build-essential. `alignair doctor` reports `derive_backend: cython`.

# ── builder: compile the optional Cython kernel into a wheel ─────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN pip install --upgrade pip build

# Everything the wheel build needs: metadata (pyproject), the Cython extension declaration (setup.py),
# the readme/license referenced by pyproject, and the sources.
COPY pyproject.toml setup.py README.md LICENSE ./
COPY src/ ./src/
RUN python -m build --wheel --outdir /wheels

# ── runtime ──────────────────────────────────────────────────────────────────
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

# Install the pre-built wheel (carries the compiled kernel) with the CLI extra.
COPY --from=builder /wheels/*.whl /tmp/wheels/
RUN WHEEL="$(ls /tmp/wheels/*.whl)" \
    && pip install "${WHEEL}[cli]" \
    && rm -rf /tmp/wheels

# Bundled example data (see examples/README.md).
COPY examples/ ./examples/

# Non-root by default.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Models are NOT baked in; they cache under $HOME. Mount a volume to persist them across runs:
#   docker run -v alignair-cache:/home/appuser/.cache/alignair ...
VOLUME ["/home/appuser/.cache/alignair"]

# Healthcheck = environment self-check (exits non-zero if a core dependency is missing).
HEALTHCHECK --interval=30s --timeout=20s --retries=3 CMD alignair doctor || exit 1

ENTRYPOINT ["alignair"]
CMD ["--help"]

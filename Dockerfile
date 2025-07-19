FROM python:3.9-slim

# Set metadata for v2.0
LABEL version="2.0.0"
LABEL description="AlignAIR v2.0 - Unified Multi-Chain IG/TCR Sequence Alignment Tool"
LABEL maintainer="Thomas Konstantinovsky & Ayelet Peres"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY checkpoints /app/pretrained_models

# Install AlignAIR v2.0
RUN pip install -e .

EXPOSE 8000

# Set environment variable for v2.0
ENV ALIGNAIR_VERSION=2.0.0

CMD ["/bin/bash"]

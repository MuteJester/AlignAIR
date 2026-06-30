#!/bin/bash
# 1M XAttnAligner-vs-IgBLAST head-to-head: build+export -> predict both -> bootstrap compare.
# Big artifacts (~30GB) live on the larger NVMe drive, not the root partition.
set -e
DIR=/media/thomas/NVMe_Data/h2h_1M
mkdir -p "$DIR"
cd /home/thomas/Desktop/AlignAIR
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

if [ ! -f "$DIR/cases.jsonl" ]; then
  log "BUILD 1M (27 strata x 37000) ..."
  $PY -m alignair.benchmark.cli build --recipe assay --n-per-stratum 37000 --n-per-focus 37000 \
      --seed 7 --workers 12 --out "$DIR/cases.jsonl" \
      --export-dir "$DIR" --export-prefix m1 --export-frame presented
fi
log "cases: $(wc -l < "$DIR/cases.jsonl")"

log "PREDICT (IgBLAST + XAttnAligner) ..."
$PY scripts/run_h2h_xattn.py --export-dir "$DIR" --prefix m1 --out "$DIR" \
    --model .private/models/xattn_igh.pt --batch-size 64 --threads 12

log "COMPARE (bootstrap 500, bonferroni) ..."
$PY -m alignair.benchmark.cli compare --cases "$DIR/cases.jsonl" \
    --a-predictions "$DIR/igblast_airr.tsv" --a-prediction-format airr-tsv \
    --b-predictions "$DIR/xattn_predictions.jsonl" --b-prediction-format jsonl \
    --model-a-name igblast --model-b-name xattn \
    --policy igh_allele_calling_core --bootstrap 500 --confidence 0.95 \
    --multiple-comparison-correction bonferroni --out "$DIR/igblast_vs_xattn_1M.json"
log "DONE -> $DIR/igblast_vs_xattn_1M.json"

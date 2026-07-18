#!/usr/bin/env sh
# Release smoke test — the documented first-run path, end to end, with NO published
# model and NO network. It runs against whatever `alignair` is on PATH, so the SAME
# script verifies a freshly-installed wheel AND the Docker image (see ci.yml).
#
#   scripts/release_smoke.sh [OUTPUT_DIR]
#
# Set ALIGNAIR to override the command (e.g. "python -m alignair.cli").
set -eu

OUT="${1:-./smoke_out}"
ALIGNAIR="${ALIGNAIR:-alignair}"

echo "== version ==";        $ALIGNAIR --version
echo "== doctor ==";         $ALIGNAIR doctor
echo "== demo (train tiny -> predict -> validate -> dynamic genotype) =="
$ALIGNAIR demo --steps 1 -o "$OUT" --device cpu
echo "== validate-airr ==";  $ALIGNAIR validate-airr "$OUT/demo.tsv"
echo "== compare (default vs donor reference, same reads) =="
$ALIGNAIR compare --a "$OUT/demo.tsv" --b "$OUT/demo_donor.tsv" \
    --a-name default-ref --b-name donor-ref --out "$OUT/agreement.md"

# Assert the expected artifacts exist and are non-empty.
for f in "$OUT/demo.tsv" "$OUT/demo_donor.tsv" "$OUT/agreement.md"; do
    if [ ! -s "$f" ]; then
        echo "SMOKE FAIL: missing or empty artifact: $f" >&2
        exit 1
    fi
done
if [ ! -d "$OUT/bundle" ]; then
    echo "SMOKE FAIL: demo did not produce a model bundle at $OUT/bundle" >&2
    exit 1
fi

echo "SMOKE OK -> $OUT"

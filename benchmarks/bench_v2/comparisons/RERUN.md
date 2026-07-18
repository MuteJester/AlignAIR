# Verified v3.0.0 head-to-head - rerun runbook

Re-run **both tools against the same frozen `../cases/`** (do not regenerate the cases - they are
canonical, checksummed ground truth) with the **released** AlignAIR model and a **recorded** IgBLAST
version, capturing full provenance. This produces the v3-attributable comparison that the docs will
cite once it exists.

## Target layout

```
comparisons/
  alignair-3.0.0_vs_igblast-<version>/
    alignair_predictions.tsv      # raw AlignAIR output, unmodified
    alignair.run.json             # AlignAIR run-provenance sidecar
    alignair_report.json          # scored AlignAIR report
    igblast_predictions.tsv       # raw IgBLAST -outfmt 19 output, unmodified
    igblast_report.json           # scored IgBLAST report
    comparison_report.json        # the paired comparison (model_comparison_report)
    provenance.json               # everything in "Provenance to capture" below
    commands.sh                   # the exact commands actually run
    SHA256SUMS                    # sha256 of every file in this directory
```

## AlignAIR half (runnable anywhere the released package + model are installed)

```bash
CASES=benchmarks/bench_v2/cases
OUT=benchmarks/bench_v2/comparisons/alignair-3.0.0_vs_igblast-<version>
mkdir -p "$OUT"

# 1. predict with the PINNED released model (records .run.json provenance next to the output)
alignair predict --input "$CASES/igh.fasta" --out "$OUT/alignair_predictions.tsv" \
  --model alignair-igh-human@1.0.0
cp "$OUT/alignair_predictions.tsv.run.json" "$OUT/alignair.run.json"   # or whatever the sidecar is named

# 2. score against the frozen manifest
PYTHONPATH=src python -m alignair_benchmark.cli evaluate \
  --predictions "$OUT/alignair_predictions.tsv" ... --out "$OUT/alignair_report.json"
```

## IgBLAST half (needs IgBLAST installed / a pinned container)

```bash
# record the version + the exact germline the model uses (export it from the model)
igblastn -version | tee "$OUT/igblast_version.txt"
alignair reference export alignair-igh-human@1.0.0 --fasta "$OUT/germline.fasta"

makeblastdb -parse_seqids -dbtype nucl -in "$OUT/germline_v.fasta" -out "$OUT/db/V"   # V, D, J
igblastn -germline_db_V "$OUT/db/V" -germline_db_D "$OUT/db/D" -germline_db_J "$OUT/db/J" \
  -auxiliary_data <aux> -organism human -ig_seqtype Ig -outfmt 19 \
  -query "$CASES/igh.fasta" > "$OUT/igblast_predictions.tsv"

PYTHONPATH=src python -m alignair_benchmark.cli normalize-predictions \
  --input "$OUT/igblast_predictions.tsv" --out "$OUT/igblast.jsonl"
PYTHONPATH=src python -m alignair_benchmark.cli evaluate \
  --predictions "$OUT/igblast.jsonl" ... --out "$OUT/igblast_report.json"
```

## Compare + finalize

```bash
PYTHONPATH=src python -m alignair_benchmark.cli compare \
  --a "$OUT/alignair_report.json" --b "$OUT/igblast_report.json" \
  --a-name AlignAIR --b-name IgBLAST --out "$OUT/comparison_report.json"

# validate + checksum
PYTHONPATH=src python -m alignair_benchmark.cli validate-artifact \
  --path "$OUT/comparison_report.json" --kind model_comparison_report
( cd "$OUT" && sha256sum * */* 2>/dev/null > SHA256SUMS )
```

(Exact `evaluate`/`build` flags: see `python -m alignair_benchmark.cli <command> --help`.)

## Provenance to capture (write into `provenance.json`)

- AlignAIR: exact release model **file path + SHA-256**; model **id/version**; **allele-order
  fingerprint** and embedded **reference hashes** (from the model card); exact **git commit** and
  **package version**; full `alignair doctor` output; environment / package versions.
- IgBLAST: **version** and the executable or **container digest**; the `makeblastdb` and `igblastn`
  commands verbatim; **auxiliary / internal-data** identity; **checksums** of every germline input and
  database file.
- Confirmation that both tools used the **same compatible allele catalog** (the exported germline).
- Complete build / prediction / comparison commands (mirror them into `commands.sh`).
- Case and result **counts**, and any **failures / exclusions** per tool.

## After the rerun

Regenerate the Benchmarks page, the design summary, and the README **from
`comparison_report.json`** - or add a test that asserts every published number and verdict matches
it. Hand-copied numbers must not be the long-term source of truth. Only then re-enable the
quantitative head-to-head (and, if wanted, the widget) on launch-facing pages.

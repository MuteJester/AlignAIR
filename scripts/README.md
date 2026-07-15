# scripts

Maintainer and research helpers. Run with `PYTHONPATH=src .venv/bin/python scripts/<name> ...`. These
are not part of the installed package — the user-facing entry point is the `alignair` CLI.

## Release / maintainer

- `build_pretrained_registry.py` — build the pretrained-model registry directory from local
  `.alignair` artifacts (re-card + validate), ready to upload to Hugging Face. See
  [`docs/publishing_models.md`](../docs/publishing_models.md).
- `publish_model.py` — publish a single pickle-free `.alignair` into a local registry directory.
- `release_smoke.sh` — end-to-end smoke (`doctor` → `demo` → `validate-airr` → `compare`) run by CI on
  the built wheel and inside the Docker image.
- `train_alignair.py` — training driver used to produce the pretrained models.
- `calibrate_model.py` — fit and embed optional per-allele confidence calibration into a model.

## Research / benchmark drivers

Ad-hoc drivers for evaluation and profiling (require data/tools under the gitignored `.private/`):
`bench_alignair_h2h.py`, `run_1m_h2h.sh`, `baseline_igblast.py` (IgBLAST baseline),
`aligner_microbench.py`, `diagnose_alignair.py`, `diag_d_ambiguity.py`.

The canonical, reproducible evaluator is the `alignair_benchmark` package (and the `alignair benchmark`
CLI command), not these drivers.

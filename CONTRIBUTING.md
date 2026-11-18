# Contributing to AlignAIR

Thanks for your interest! AlignAIR is a neural IG/TCR aligner; contributions of bug reports,
references/models, and code are all welcome.

## Development setup

```bash
git clone https://github.com/MuteJester/AlignAIR && cd AlignAIR
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
python -m pytest tests      # source suite: the product, the assay and the architecture lab
alignair doctor             # check your environment
```

Release CI uses `sh tests/packaging/release_smoke.sh OUTPUT_DIR` against an installed package and
inside the Docker image. `src/` also holds two development packages that are excluded from the wheel:
`alignair_assay`, which scores AlignAIR and IgBLAST against simulated truth and real donors (its
[README](src/alignair_assay/README.md) covers its workflows, including freezing GenAIRR first), and
`alignair_lab`, which runs architecture experiments.
The [testing guide](https://mutejester.github.io/AlignAIR/#/docs/testing) explains suite ownership,
test selection, optional local fixtures, and the research test dependencies included in `[dev]`.
Required public-model and release-claim checks use checksum-pinned fixtures and the separate model
acceptance workflow. The guide includes the equivalent local command. Declare each new test's
`component`, `integration`, or `acceptance` layer and use `tmp_path` for generated outputs.

## Pull requests

- Open an issue first for non-trivial changes so we can agree on the approach.
- Keep PRs focused; include tests for new behavior (`pytest` must pass).
- Match the surrounding code style; no large unrelated reformatting.
- Update docs / `CHANGELOG.md` when you change user-facing behavior.

## Reporting issues

For exception classes, structured diagnostics, CLI error reporting, and conventions
for new failure paths, see the [error codes](https://mutejester.github.io/AlignAIR/#/docs/error-codes).

Use the issue templates: **bug report**, **feature request**, or **model or reference request** (a
species/locus you'd like a pretrained model for, or a reference that fails to parse or train). Include your
`alignair doctor` output and, for prediction issues, a minimal input file.

## Scope

Code is GPL-3.0-or-later. By contributing you agree your contribution is licensed under the same
terms. Pretrained models and reference data may carry their own licenses — note these in model
cards / PRs.

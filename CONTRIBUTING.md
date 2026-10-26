# Contributing to AlignAIR

Thanks for your interest! AlignAIR is a neural IG/TCR aligner; contributions of bug reports,
references/models, and code are all welcome.

## Development setup

```bash
git clone https://github.com/MuteJester/AlignAIR && cd AlignAIR
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
python -m pytest tests      # source suite, including AA research tests
alignair doctor             # check your environment
```

Release CI uses `sh tests/packaging/release_smoke.sh OUTPUT_DIR` against an installed package and
inside the Docker image. The [AA guard research API guide](https://mutejester.github.io/AlignAIR/#/docs/aa-guard-research)
documents the package commands for collection, cohort auditing, profile construction, and training studies.
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
for new failure paths, see the [Errors and diagnostics API](https://mutejester.github.io/AlignAIR/#/docs/errors).

Use the issue templates: **bug report**, **feature request**, **model request** (a species/locus
you'd like a pretrained model for), or **reference parsing / training failure**. Include your
`alignair doctor` output and, for prediction issues, a minimal input file.

## Scope

Code is GPL-3.0-or-later. By contributing you agree your contribution is licensed under the same
terms. Pretrained models and reference data may carry their own licenses — note these in model
cards / PRs.

# Contributing to AlignAIR

Thanks for your interest! AlignAIR is a neural IG/TCR aligner; contributions of bug reports,
references/models, and code are all welcome.

## Development setup

```bash
git clone https://github.com/MuteJester/AlignAIR && cd AlignAIR
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
pytest                      # run the test suite
alignair doctor             # check your environment
```

## Pull requests

- Open an issue first for non-trivial changes so we can agree on the approach.
- Keep PRs focused; include tests for new behavior (`pytest` must pass).
- Match the surrounding code style; no large unrelated reformatting.
- Update docs / `CHANGELOG.md` when you change user-facing behavior.

## Reporting issues

Use the issue templates: **bug report**, **feature request**, **model request** (a species/locus
you'd like a pretrained model for), or **reference parsing / training failure**. Include your
`alignair doctor` output and, for prediction issues, a minimal input file.

## Scope

Code is GPL-3.0-or-later. By contributing you agree your contribution is licensed under the same
terms. Pretrained models and reference data may carry their own licenses — note these in model
cards / PRs.

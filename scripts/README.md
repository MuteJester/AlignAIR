# scripts

Public helper scripts. Run with `PYTHONPATH=src .venv/bin/python scripts/<name> ...`. These are not
part of the installed package — the user-facing entry point is the `alignair` CLI.

- `release_smoke.sh` — end-to-end release smoke (`doctor` → `demo` → `validate-airr` → `compare`),
  run by CI on the built wheel and inside the Docker image.

Maintainer-only tooling (model publishing, calibration, diagnostics) lives outside the public repo,
in the gitignored `.private/scripts/`. Publishing to the `AlignAIR/AlignAIR-pretrained` hub requires a
HuggingFace write token for the AlignAIR org — nothing in this repo can push to it.

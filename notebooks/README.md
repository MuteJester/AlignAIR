# Notebooks

Two runnable notebooks for evaluating AlignAIR. Both are **executed in the test suite**
(`tests/alignair/test_notebooks.py`) so they don't go stale. Run them **from the repo root**.

| Notebook | What it shows |
|----------|---------------|
| [`01_inference.ipynb`](01_inference.ipynb) | The inference workflow via the public Python API (`load_model` → `predict` → `to_airr`), incl. the dynamic-genotype path. Uses a tiny `alignair demo` model as a stand-in until a pretrained bundle is published. |
| [`02_custom_reference_training.ipynb`](02_custom_reference_training.ipynb) | Train on your own germline FASTA (`alignair train`): discover references, `--plan`, a smoke train, then predict with the custom-reference bundle. |

```bash
pip install "AlignAIR[cli]"        # plus jupyter to run interactively
jupyter lab                        # then open notebooks/
```

The notebooks call the CLI as `!{sys.executable} -m alignair.cli ...` so they work whether or
not the `alignair` console script is on `PATH`. **Colab**: they run on Colab once AlignAIR is
published to PyPI (`pip install "AlignAIR[cli]"`); until then install from a local checkout.

> Both notebooks use tiny/few-step models, so their **calls are not accurate** — they
> demonstrate the workflow. Train `--preset desktop`/`standard` for real results.

<p align="center">
  <img src="https://alignair.ai/_next/static/media/logo_alignair11bw.17e5d8d6.svg" width="220" alt="AlignAIR logo" />
</p>

<h1 align="center">AlignAIR</h1>
<p align="center">
  Alignment and allele calling for immunoglobulin (IG) and T‑cell receptor (TCR) repertoires.<br>
  <a href="https://hub.docker.com/r/thomask90/alignair"><img alt="Docker pulls" src="https://img.shields.io/docker/pulls/thomask90/alignair"></a>
  <a href="https://doi.org/10.5281/zenodo.15687939"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15687939.svg" alt="DOI"></a>
  <a href="LICENSE"><img alt="GPLv3" src="https://img.shields.io/badge/license-GPLv3-blue.svg"></a>
</p>

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Selecting a Model](#selecting-a-model)
3. [Model Bundles](#model-bundles)
4. [Running Predictions](#running-predictions)
5. [Multi-Chain Details](#multi-chain-details)
6. [SavedModel Export & Fine-Tuning](#savedmodel-export--fine-tuning)
7. [Parameters Reference](#parameters-reference)
8. [Docker (Advanced)](#docker-advanced)
9. [Development & Contribution](#development--contribution)
10. [Data & Citation](#data--citation)
11. [License](#license)
12. [Contact](#contact)

---

## Quick Start

### A. Docker

Pull image:
```bash
docker pull thomask90/alignair:latest
```

List bundled pretrained models:
```bash
docker run --rm -it thomask90/alignair:latest list-pretrained
```
Example output:
```
Bundle                 Type           SeqLen   Chains                             Status
IGH_S5F_576            single_chain   576      -                                  OK
IGH_S5F_576_Extended   single_chain   576      -                                  OK
IGL_S5F_576            multi_chain    576      BCR_LIGHT_LAMBDA,BCR_LIGHT_KAPPA   OK
TCRB_UNIFORM_576       single_chain   576      -                                  OK
```

Optional flags:
```bash
docker run --rm -it thomask90/alignair:latest list-pretrained --show-files
docker run --rm -it thomask90/alignair:latest list-pretrained --json-output
```

Run heavy chain (extended):
```bash
docker run --rm -v /path/to/input:/data -v /path/to/output:/out \
  thomask90/alignair:latest run \
  --model-dir=/app/pretrained_models/IGH_S5F_576_Extended \
  --genairr-dataconfig=HUMAN_IGH_EXTENDED \
  --sequences=/data/sequences.csv \
  --save-path=/out \
  --translate-to-asc
```

Windows (PowerShell) path example:
```powershell
docker run --rm `
  -v C:/Users/you/Datasets:/data `
  -v C:/Users/you/Downloads:/out `
  thomask90/alignair:latest run `
  --model-dir=/app/pretrained_models/IGH_S5F_576_Extended `
  --genairr-dataconfig=HUMAN_IGH_EXTENDED `
  --sequences=/data/sequences.csv `
  --save-path=/out
```

Output file: `/out/<input_basename>_alignairr_results.csv`

### B. Local (Editable Install)

```bash
git clone https://github.com/MuteJester/AlignAIR.git
cd AlignAIR
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python app.py list-pretrained --root checkpoints
python app.py run --model-dir=checkpoints/IGH_S5F_576 --sequences=tests/data/test/sample_igh_extended.csv --save-path=tmp_out
```
Python requirement: >=3.9,<3.12.

---

## Selecting a Model

| Bundle | Type | Max Seq Len | Chains (multi) | Path |
|--------|------|-------------|----------------|------|
| IGH_S5F_576 | single_chain | 576 | - | `/app/pretrained_models/IGH_S5F_576` |
| IGH_S5F_576_Extended | single_chain | 576 | - | `/app/pretrained_models/IGH_S5F_576_Extended` |
| IGL_S5F_576 | multi_chain | 576 | Lambda,Kappa | `/app/pretrained_models/IGL_S5F_576` |
| TCRB_UNIFORM_576 | single_chain | 576 | - | `/app/pretrained_models/TCRB_UNIFORM_576` |

Choose:
- Standard heavy chain: `IGH_S5F_576`
- Extended heavy chain: `IGH_S5F_576_Extended`
- Lambda + Kappa (classification): `IGL_S5F_576`
- TCR beta: `TCRB_UNIFORM_576`

List programmatically:
```bash
docker run --rm thomask90/alignair:latest list-pretrained
```

---

## Model Bundles

Bundle layout:
```
model_dir/
  config.json
  dataconfig.pkl
  training_meta.json        # optional
  VERSION
  fingerprint.txt
  saved_model/
  checkpoint.weights.h5     # optional (fine‑tuning)
  README.md                 # optional
```

Why bundles:
- Structural + dataconfig reproducibility
- Integrity check via fingerprint
- Single directory: metadata + SavedModel (+ optional weights)

Legacy non‑bundle checkpoints still load; prefer bundles for new work.

Create during training (example):
```python
trainer.train(..., save_pretrained=True)
```

Load in Python:
```python
from AlignAIR.Models.SingleChainAlignAIR.SingleChainAlignAIR import SingleChainAlignAIR
model = SingleChainAlignAIR.from_pretrained('path/to/bundle')
```

CLI:
```bash
python app.py run --model-dir=path/to/bundle --genairr-dataconfig=HUMAN_IGH_OGRDB --sequences=input.csv --save-path=out
```

Integrity: mismatch raises on load if fingerprint differs.

Migrate legacy:
```python
legacy_model.save_pretrained('new_bundle_dir')
```

---

## Running Predictions

```bash
python app.py run \
  --model-dir=checkpoints/IGH_S5F_576 \
  --genairr-dataconfig=HUMAN_IGH_OGRDB \
  --sequences=tests/data/test/sample_igh_extended.csv \
  --save-path=tmp_out
```

Output naming: `<input_basename>_alignairr_results.csv` in `--save-path`.

Threshold application: probabilities filtered by threshold; caps applied afterward.

Optional flags:
- `--translate-to-asc` for ASC allele labels
- `--airr-format` for AIRR schema output

---

## Multi-Chain Details

Triggered when `--genairr-dataconfig` has >1 comma‑separated entry.

Example (Lambda + Kappa):
```
--genairr-dataconfig=HUMAN_IGL_OGRDB,HUMAN_IGK_OGRDB
```
Ordering must match training (Lambda first). Output includes `chain_type`.

Single chain: supply one identifier or path to a dataconfig pickle.

---

## SavedModel Export & Fine-Tuning

SavedModel is under `saved_model/` inside a bundle.

Export (via bundle creation):
```python
model.save_pretrained('bundle_dir')
```

Direct export:
```python
model.export_saved_model('export_dir/saved_model')
```

Load SavedModel:
```python
import tensorflow as tf
sm = tf.saved_model.load('bundle_dir/saved_model')
serving_fn = sm.signatures['serving_default']
```

Bundle vs SavedModel:
- General use & metadata: bundle path
- Serving stack: `saved_model/`
- Fine‑tuning: rebuild model + load `checkpoint.weights.h5` if present

Fine‑tuning steps:
```python
from pathlib import Path
from AlignAIR.Serialization.io import load_bundle
from AlignAIR.Models.SingleChainAlignAIR.SingleChainAlignAIR import SingleChainAlignAIR
cfg, dataconfig, meta = load_bundle(Path('bundle'))
model = SingleChainAlignAIR(max_seq_length=cfg.max_seq_length, dataconfig=dataconfig)
if (Path('bundle') / 'checkpoint.weights.h5').exists():
    model.load_weights('bundle/checkpoint.weights.h5').expect_partial()
```

---

## Parameters Reference

### Core
| Flag | Description | Default |
|------|-------------|---------|
| `--model-dir` | Model bundle directory | (required) |
| `--model-checkpoint` | Legacy checkpoint directory | None |
| `--genairr-dataconfig` | Built‑in name(s) or path(s); comma‑separated for multi-chain | HUMAN_IGH_OGRDB |
| `--sequences` | Input CSV/TSV/FASTA | (required) |
| `--save-path` | Output directory | (required) |
| `--batch-size` | Batch size | 2048 |

### Threshold / Caps
| Flag | Description | Default |
|------|-------------|---------|
| `--v-allele-threshold` | V allele threshold | 0.75 |
| `--d-allele-threshold` | D allele threshold | 0.30 |
| `--j-allele-threshold` | J allele threshold | 0.80 |
| `--v-cap` / `--d-cap` / `--j-cap` | Max retained alleles | 3 |

### Output / Format
| Flag | Description | Default |
|------|-------------|---------|
| `--translate-to-asc` | Translate allele names | False |
| `--airr-format` | AIRR schema output | False |
| `--fix-orientation` | Orientation correction | True |

### Misc
| Flag | Description |
|------|-------------|
| `--config-file` | YAML with parameters |
| `--custom-orientation-pipeline-path` | Custom orientation pipeline |
| `--custom-genotype` | Genotype file for likelihood adjustment |
| `--save-predict-object` | Persist internal object (debug) |

Full help: `docker run thomask90/alignair:latest run --help`

---

## Docker (Advanced)

Custom bundle:
```bash
docker run --rm -v /models/my_bundle:/bundle -v /data:/data -v /out:/out \
  thomask90/alignair:latest run \
  --model-dir=/bundle \
  --sequences=/data/sequences.csv \
  --genairr-dataconfig=HUMAN_IGH_OGRDB \
  --save-path=/out
```

List mounted directory:
```bash
docker run --rm -v /models:/extra thomask90/alignair:latest list-pretrained --root /extra --json-output > bundles.json
```

JSON inspection:
```bash
docker run --rm thomask90/alignair:latest list-pretrained --json-output | jq '.[].name'
```

Windows paths: prefer forward slashes; ensure drive sharing is enabled.

---

## Development & Contribution

Quick commands:
```bash
pip install -e .
pytest -q
python app.py list-pretrained --root checkpoints
python app.py run --model-dir=checkpoints/IGH_S5F_576 --sequences=tests/data/test/sample_igh_extended.csv --save-path=tmp_out
```

Workflow:
1. Branch
2. Add/update tests
3. `pytest -q`
4. Open PR

License: GPLv3 (see `LICENSE`).

---

## Data & Citation

Citation DOI:
```
doi:10.5281/zenodo.15687939
```

---

## License

GPL v3.0 or later (see `LICENSE`).

---

## Contact

Issues: GitHub issues
Email: thomaskon90@gmail.com
Site: https://alignair.ai

<p align="center">
  <img src="https://alignair.ai/_next/static/media/logo_alignair11bw.17e5d8d6.svg" width="240" alt="AlignAIR logo">
</p>

<h1 align="center">AlignAIR</h1>
<p align="center">
  <strong>Deep‑learning sequence aligner for immunoglobulin &amp; T‑cell receptor repertoires</strong><br>
  <a href="https://hub.docker.com/r/thomask90/alignair">
    <img alt="Docker pulls" src="https://img.shields.io/docker/pulls/thomask90/alignair">
  </a>
<a href="https://doi.org/10.5281/zenodo.15687939"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15687939.svg" alt="DOI"></a>
<a href="LICENSE">
  <img alt="GPLv3" src="https://img.shields.io/badge/license-GPLv3-blue.svg">
</a>
</p>
 
---

### Quick Start

**Step 1: Pull the Docker image**
```bash
docker pull thomask90/alignair:latest
```

**Step 2: Start container with your data volumes**
```bash
# Mount your input data and output directories
docker run -it --rm \
  -v /path/to/your/input/data:/data \
  -v /path/to/your/output/downloads:/downloads \
  thomask90/alignair:latest
```

**Step 3: Run AlignAIR inside the container**
```bash
# Example: Heavy Chain analysis with extended model
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGH_S5F_576_EXTENDED \
  --genairr-dataconfig=HUMAN_IGH_EXTENDED \
  --sequences=/data/your_sequences.csv \
  --save-path=/downloads/
```

<details>
<summary>Table of contents</summary>

- [Key features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Pretrained Bundles](#pretrained-bundles)
- [SavedModel Export](#savedmodel-export)
- [Available Models](#available-models)
- [Docker in depth](#docker-in-depth)
- [Examples](#examples)
- [Parameter Reference](#parameter-reference)
- [Data availability](#data-availability)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

</details>

---

## What's New in v2.0

AlignAIR v2.0 introduces a revolutionary unified architecture:

### Unified Models
- **SingleChainAlignAIR**: Optimized for single receptor type analysis
- **MultiChainAlignAIR**: Native multi-chain support with chain type classification
- **Universal compatibility**: Works with any GenAIRR dataconfig combination

### Multi-Chain Analysis
- **Mixed receptor processing**: Analyze IGK + IGL light chains simultaneously
- **Chain type classification**: Automatic receptor type identification
- **Optimized batch processing**: Equal partitioning across chain types

### Dynamic GenAIRR Integration
- **Built-in dataconfigs**: `HUMAN_IGH_OGRDB`, `HUMAN_IGK_OGRDB`, `HUMAN_IGL_OGRDB`, `HUMAN_TCRB_IMGT`
- **Custom config support**: Use your own GenAIRR dataconfigs
- **Automatic detection**: Single vs. multi-chain mode based on input

### Enhanced Performance
- **Streamlined architecture**: Single codebase for all receptor types
- **Memory optimization**: Efficient processing for large datasets
- **GPU acceleration**: Optimized tensor operations

---

## Key features
- **State‑of‑the‑art accuracy** for V, D, J allele calling and junction segmentation  
- **Unified multi‑chain architecture** supporting any chain combinations with dynamic GenAIRR integration
- **Multi‑task deep network** jointly optimises alignment, productivity, indel detection, and chain type classification  
- **Scales to millions** of AIRR‑seq reads with GPU support  
- **Universal model architecture** that adapts to single-chain or multi-chain scenarios
- **Dynamic data configuration** with built-in GenAIRR dataconfigs for major species and receptors
- **Drop‑in integration** with AIRR schema & downstream tools

---

## Installation

### Docker (recommended)

```bash
# Pull the latest image
docker pull thomask90/alignair:latest

# Start interactive container (mount local data to /data)
docker run -it --rm -v /path/to/local/data:/data thomask90/alignair:latest
```

> **Prerequisites:** Nvidia GPU + CUDA 11 recommended (CPU works, slower).

### Local (advanced)

```bash
git clone https://github.com/MuteJester/AlignAIR.git
cd AlignAIR && pip install -e ./
```
* Note that the local version comes without pretrained model weights and is mainly
used for custom model and pipeline development, testing, and debugging.
It is mainly recommended for developers, contributors and advanced users.

---

## Usage

### Basic Usage

```bash
python app.py run \
    --model-checkpoint=/app/pretrained_models/IGH_S5F_576 \
    --genairr-dataconfig=HUMAN_IGH_OGRDB \
    --sequences=/data/input/sequences.csv \
    --save-path=/data/output
```

---

## Pretrained Bundles

AlignAIR now supports a reproducible, versioned bundle format that packages:

```
model_dir/
  config.json            # Structural model configuration (architecture & allele counts)
  dataconfig.pkl         # GenAIRR (or MultiDataConfigContainer) object used to build the model
  weights.h5             # Keras weights
  training_meta.json     # Optional: epoch counts, final metrics, optimizer, lr, notes
  VERSION                # Serialization format version (e.g. 1)
  fingerprint.txt        # SHA256 over critical artifacts (integrity check)
  README.md (optional)   # User notes
  saved_model/ (optional) # TensorFlow SavedModel (if exported)
```

### Why Bundles?
- No need to manually reconstruct model parameters.
- Guards against dataconfig / allele count mismatches.
- Self-describing, portable, and integrity‑verifiable.
- Backward compatible: you can still point to legacy checkpoints (weights directories).

### Creating a Bundle During Training
Use the Python Trainer (example snippet):

```python
from src.AlignAIR.Trainers import Trainer
trainer = Trainer(model, session_path='./runs', model_name='HeavyExample')
trainer.train(train_dataset, epochs=3, samples_per_epoch=1024, batch_size=32,
              save_pretrained=True,  # turns on bundle export
              export_saved_model=True,  # also produce SavedModel
              include_logits_in_saved_model=False,
              training_notes='Baseline heavy chain experiment')
```

If `bundle_dir` is not provided it defaults to `./runs/HeavyExample_bundle`.

### Loading a Bundle in Python
```python
from AlignAIR.Models.SingleChainAlignAIR.SingleChainAlignAIR import SingleChainAlignAIR
model = SingleChainAlignAIR.from_pretrained('path/to/HeavyExample_bundle')
preds = model({'tokenized_sequence': encoded_batch})
```

For multi-chain:
```python
from AlignAIR.Models.MultiChainAlignAIR.MultiChainAlignAIR import MultiChainAlignAIR
multi_model = MultiChainAlignAIR.from_pretrained('path/to/LightChains_bundle')
```

### Using a Bundle with the CLI
Instead of `--model_checkpoint` you can now supply:
```bash
python app.py run \
  --model_dir=/path/to/HeavyExample_bundle \
  --genairr-dataconfig=HUMAN_IGH_OGRDB \
  --sequences=/data/input.csv \
  --save-path=/data/out
```
If both `--model_dir` and `--model_checkpoint` are provided, `--model_dir` takes precedence.

### Integrity Verification
Every bundle includes a `fingerprint.txt` (SHA256). If files are altered, fingerprint validation during `from_pretrained` will raise an error.

### Migration from Legacy Checkpoints
Legacy directories **without** `config.json` are still supported (legacy flow). For reproducibility, re‑export them:
```python
legacy_model = SingleChainAlignAIR(max_seq_length=576, dataconfig=...)  # build as before
legacy_model.load_weights('old_weights_dir').expect_partial()
legacy_model.save_pretrained('new_bundle_dir')
```

---

## SavedModel Export

Bundles can optionally embed a TensorFlow SavedModel (subdirectory `saved_model/`) for deployment in serving stacks (TF Serving, Triton, etc.).

### Export During Training
Use the Trainer flags (see above) or call directly:
```python
model.save_pretrained('bundle_dir', export_saved_model=True, include_logits_in_saved_model=False)
```

### Stand‑Alone Export
```python
model.export_saved_model('export_dir/saved_model', include_logits=False)
```

### Loading a SavedModel
```python
import tensorflow as tf
sm = tf.saved_model.load('bundle_dir/saved_model')
serving_fn = sm.signatures['serving_default']
outputs = serving_fn(tokenized_sequence=tf.constant(batch_ids, dtype=tf.int32))
print(outputs.keys())  # e.g. dict_keys(['v_start','v_end','j_start','j_end','v_allele','j_allele',...])
```

### Included Outputs
Single chain (base): `v_start`, `v_end`, `j_start`, `j_end`, `v_allele`, `j_allele`, `mutation_rate`, `indel_count`, `productive` (+ D‑gene outputs if applicable). Multi-chain adds `chain_type`.

Set `include_logits_in_saved_model=True` (or `include_logits=True` for direct export) to append raw boundary logits (`*_start_logits`, `*_end_logits`).

### When to Use SavedModel vs. Bundles?
| Scenario | Use Bundle | Use SavedModel |
|----------|------------|----------------|
| Research reproducibility | ✅ | optional |
| Fine‑tuning / further training | ✅ | ❌ (graph only) |
| Production inference (serving stack) | ✅ (for metadata) + SavedModel | ✅ |
| Integrity & config inspection | ✅ | limited |

---

### Example Commands

**Heavy Chain Analysis (Extended Model):**
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGH_S5F_576_EXTENDED \
  --genairr-dataconfig=HUMAN_IGH_EXTENDED \
  --sequences=/data/heavy_sequences.csv \
  --save-path=/downloads/ \
  --v-allele-threshold=0.75 \
  --d-allele-threshold=0.3 \
  --j-allele-threshold=0.8
```

**Light Chain Multi-Chain Analysis (Lambda + Kappa with Chain Type Prediction):**
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGL_S5F_576 \
  --genairr-dataconfig=HUMAN_IGL_OGRDB,HUMAN_IGK_OGRDB \
  --sequences=/data/mixed_light_sequences.csv \
  --save-path=/downloads/ \
  --airr-format
```
> **Output includes:** Chain type prediction in `chain_type` column (Lambda or Kappa)

**Single Light Chain Analysis:**
```bash
# Lambda only (using multi-chain model)
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGL_S5F_576 \
  --genairr-dataconfig=HUMAN_IGL_OGRDB,HUMAN_IGK_OGRDB \
  --sequences=/data/lambda_sequences.csv \
  --save-path=/downloads/ \
  --airr-format \
  --fix-orientation

# Kappa only (using multi-chain model)  
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGL_S5F_576 \
  --genairr-dataconfig=HUMAN_IGL_OGRDB,HUMAN_IGK_OGRDB \
  --sequences=/data/kappa_sequences.csv \
  --save-path=/downloads/ \
  --airr-format \
  --fix-orientation
```
> **Note:** The MultiChainAlignAIR model requires both dataconfigs but will predict the correct chain type for each sequence

**T-Cell Receptor Beta Chain:**
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/TCRB_Uniform_576 \
  --genairr-dataconfig=HUMAN_TCRB_IMGT \
  --sequences=/data/tcr_sequences.csv \
  --save-path=/downloads/
```

---

## Available Models and Configurations

AlignAIR v2.0 introduces a unified architecture that dynamically adapts to different chain types and configurations using GenAIRR dataconfigs:

### Model Architecture Types

| Architecture | Use Case | DataConfig Support | Multi-Chain |
|--------------|----------|-------------------|-------------|
| **SingleChainAlignAIR** | Single receptor type analysis | Single GenAIRR dataconfig | No |
| **MultiChainAlignAIR** | Mixed receptor analysis | Multiple GenAIRR dataconfigs | Yes |

### Built-in GenAIRR DataConfigs

| DataConfig | Chain Type | Species | Reference | D Gene | Model Compatibility |
|------------|------------|---------|-----------|--------|--------------------|
| `HUMAN_IGH_OGRDB` | Heavy Chain | Human | OGRDB | ✓ | IGH_S5F_576 |
| `HUMAN_IGH_EXTENDED` | Heavy Chain Extended | Human | OGRDB + Custom | ✓ | IGH_S5F_576_EXTENDED |
| `HUMAN_IGK_OGRDB` | Kappa Light | Human | OGRDB | ✗ | IGL_S5F_576 (multi-chain) |
| `HUMAN_IGL_OGRDB` | Lambda Light | Human | OGRDB | ✗ | IGL_S5F_576 (multi-chain only) |
| `HUMAN_TCRB_IMGT` | TCR Beta | Human | IMGT  | ✓ | TCRB_Uniform_576 |

### Pre-trained Model Checkpoints

The Docker container ships with optimized models for common use cases:

| Model | Architecture | Supported Configs | Checkpoint Path | Use Case |
|-------|-------------|-------------------|-----------------|----------|
| **Heavy Chain Extended** | SingleChainAlignAIR | `HUMAN_IGH_EXTENDED` | `/app/pretrained_models/IGH_S5F_576_EXTENDED` | Enhanced heavy chain with extended allele coverage |
| **Heavy Chain Standard** | SingleChainAlignAIR | `HUMAN_IGH_OGRDB` | `/app/pretrained_models/IGH_S5F_576` | Standard heavy chain analysis |
| **Multi-Light** | MultiChainAlignAIR | `HUMAN_IGL_OGRDB,HUMAN_IGK_OGRDB` | `/app/pretrained_models/IGL_S5F_576` | Lambda + Kappa analysis with chain type prediction |
| **TCR Beta** | SingleChainAlignAIR | `HUMAN_TCRB_IMGT` | `/app/pretrained_models/TCRB_Uniform_576` | T-cell receptor beta chain |

> **Note:** The Multi-Light model (`IGL_S5F_576`) is a MultiChainAlignAIR instance that requires both Lambda and Kappa dataconfigs and always outputs chain type predictions.

### Custom DataConfigs

You can use custom GenAIRR dataconfigs by providing a path to a pickled DataConfig object:

```bash
python app.py run \
  --model-checkpoint=path/to/custom/model \
  --genairr-dataconfig=/path/to/custom_dataconfig.pkl \
  --sequences=input.csv \
  --save-path=output/
```

For multi-chain custom configs:
```bash
python app.py run \
  --model-checkpoint=path/to/multichain/model \
  --genairr-dataconfig=/path/to/config1.pkl,/path/to/config2.pkl \
  --sequences=input.csv \
  --save-path=output/
```

---

## Docker in depth

### Step-by-Step Docker Usage Guide

**1. Pull the latest AlignAIR image**
```bash
docker pull thomask90/alignair:latest
```

**2. Prepare your data**
- Ensure your input sequences are in CSV format with a `sequence` column
- Create directories for input data and output results

**3. Start the container with volume mounts**
```bash
# Windows example:
docker run -it --rm \
  -v C:/path/to/your/data:/data \
  -v C:/path/to/your/downloads:/downloads \
  thomask90/alignair:latest

# Linux/Mac example:
docker run -it --rm \
  -v /path/to/your/data:/data \
  -v /path/to/your/downloads:/downloads \
  thomask90/alignair:latest
```

**4. Run AlignAIR with the appropriate model**

#### Heavy Chain Analysis (Extended Model)
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGH_S5F_576_EXTENDED \
  --genairr-dataconfig=HUMAN_IGH_EXTENDED \
  --sequences=/data/sample_HeavyChain_dataset.csv \
  --save-path=/downloads/
```

#### Light Chain Multi-Chain Analysis (Lambda + Kappa)
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGL_S5F_576 \
  --genairr-dataconfig=HUMAN_IGL_OGRDB,HUMAN_IGK_OGRDB \
  --sequences=/data/sample_LightChain_dataset.csv \
  --save-path=/downloads/
```
> **Important:** This MultiChainAlignAIR model predicts both Lambda and Kappa chains. The output includes a `chain_type` column indicating the predicted chain type for each sequence. The order of dataconfigs (Lambda first, then Kappa) must match the training order.

#### Single Light Chain Analysis
```bash
# Lambda or Kappa only (using multi-chain model with both dataconfigs)
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGL_S5F_576 \
  --genairr-dataconfig=HUMAN_IGL_OGRDB,HUMAN_IGK_OGRDB \
  --sequences=/data/light_chain_sequences.csv \
  --save-path=/downloads/
```
> **Note:** Even for single chain type analysis, the MultiChainAlignAIR model requires both dataconfigs but will correctly predict and classify each sequence's chain type.

#### T-Cell Receptor Beta Chain
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/TCRB_Uniform_576 \
  --genairr-dataconfig=HUMAN_TCRB_IMGT \
  --sequences=/data/tcr_sequences.csv \
  --save-path=/downloads/
```

### Critical Notes for Custom Models

- **Always use the same GenAIRR dataconfig** during prediction as was used during model training
- **Never use modified dataconfigs** with pre-trained models
- **For multi-chain models:** The order of dataconfigs must match the training order exactly
- **Custom dataconfigs:** Provide the path to your pickled DataConfig object instead of built-in names

### Custom DataConfig Example
```bash
python app.py run \
  --model-checkpoint=/path/to/your/custom/model \
  --genairr-dataconfig=/data/your_custom_dataconfig.pkl \
  --sequences=/data/sequences.csv \
  --save-path=/downloads/
```

**5. Check results**
Your results will be saved in the mounted `/downloads` directory and can be accessed from your host system.

---

## Parameter Reference

### Core Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model-checkpoint` | Path to model weights | Required |
| `--chain-type` | Specify heavy, light, or tcrb | Required |
| `--sequences` | Input file path (CSV/TSV/FASTA) | Required |
| `--save-path` | Output directory | Required |

### Model Settings
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max-input-size` | Maximum input window size | `576` |
| `--batch-size` | Sequences per batch | `2048` |

### Thresholds
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--v-allele-threshold` | V allele calling threshold | `0.75` |
| `--d-allele-threshold` | D allele calling threshold | `0.30` |
| `--j-allele-threshold` | J allele calling threshold | `0.80` |
| `--v-cap` / `--d-cap` / `--j-cap` | Maximum calls per segment | `3` |

### Output Options
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--airr-format` | Output full AIRR Schema | `false` |
| `--fix-orientation` | Auto-correct orientations | `true` |
| `--translate-to-asc` | Output ASC allele names | `false` |

For complete parameter list: `python app.py run --help`

---

## Examples
See the **`examples/`** folder for Jupyter notebooks:

1. End‑to‑end heavy‑chain pipeline  
2. Benchmark vs. IgBLAST on 10 K reads  
3. Batch processing workflows

---

## Data availability
Training & benchmark datasets are archived on Zenodo: `doi:10.5281/zenodo.XXXXXXXX`

---

## Documentation
For comprehensive documentation, examples, and technical details, visit:
**[https://alignair.ai/docs](https://alignair.ai/docs)**

---

## Contributing
Pull requests are welcome! Please:

1. Run `pre-commit run --all-files`
2. Ensure `pytest` passes
3. Update `CHANGELOG.md`

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## License
This project is licensed under the terms of the [GNU General Public License v3.0 or later (GPLv3+)](LICENSE).

---

## Contact
Open an issue or email **thomaskon90@gmail.com**.  
For announcements, visit <https://alignair.ai> or join our Slack.

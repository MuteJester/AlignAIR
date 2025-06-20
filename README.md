<p align="center">
  <img src="https://alignair.ai/_next/static/media/logo_alignair11bw.17e5d8d6.svg" width="240" alt="AlignAIR logo">
</p>

<h1 align="center">AlignAIR</h1>
<p align="center">
  <strong>Deep‑learning sequence aligner for immunoglobulin &amp; T‑cell receptor repertoires</strong><br>
  <a href="https://hub.docker.com/r/thomask90/alignair">
    <img alt="Docker pulls" src="https://img.shields.io/docker/pulls/thomask90/alignair">
  </a>
  <a href="https://zenodo.org/record/XXXXXXXX">
    <img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXX.svg">
  </a>
  <a href="LICENSE">
    <img alt="MIT" src="https://img.shields.io/badge/license-MIT-green.svg">
  </a>
</p>
 
---

### ✨ Quick Start

```bash
docker pull thomask90/alignair:latest
docker run -it --rm \
  -v ~/data:/data \
  thomask90/alignair:latest

# Inside the container:
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGH_S5F_576 \
  --save-path=/data/output \
  --chain-type=heavy \
  --sequences=/app/tests/sample_HeavyChain_dataset.csv
```

<details>
<summary>Table of contents</summary>

- [Key features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
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

## Key features
- **State‑of‑the‑art accuracy** for V, D, J allele calling and junction segmentation  
- **Multi‑task deep network** jointly optimises alignment, productivity, and indel detection  
- **Scales to millions** of AIRR‑seq reads with GPU support  
- **Pre-trained models** for IGH, IGL, and TCRB chains included
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
    --chain-type=heavy \
    --sequences=/data/input/sequences.csv \
    --save-path=/data/output
```

### Example Commands

**Heavy Chain Analysis:**
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGH_S5F_576 \
  --chain-type=heavy \
  --sequences=/data/input/heavy_sequences.csv \
  --save-path=/data/output/heavy_results \
  --v-allele-threshold=0.75 \
  --d-allele-threshold=0.3 \
  --j-allele-threshold=0.8
```

**Light Chain Analysis:**
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGL_S5F_576 \
  --chain-type=light \
  --sequences=/data/input/light_sequences.csv \
  --save-path=/data/output/light_results \
  --airr-format \
  --fix-orientation
```

**T-Cell Receptor Beta Chain:**
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/TCRB_S5F_576 \
  --chain-type=tcrb \
  --sequences=/data/input/tcr_sequences.csv \
  --save-path=/data/output/tcr_results
```

---

## Available Models

The Docker container ships with three pre-trained models:

| Model | Chain Type | Checkpoint Path | Use Case |
|-------|------------|-----------------|----------|
| **IGH Heavy Chain** | `heavy` | `/app/pretrained_models/IGH_S5F_576` | B-cell heavy chain analysis |
| **IGL Light Chain** | `light` | `/app/pretrained_models/IGL_S5F_576` | B-cell lambda light chain analysis |
| **TCRB Beta Chain** | `tcrb` | `/app/pretrained_models/TCRB_S5F_576` | T-cell receptor beta chain analysis |

All models are trained on human sequences using IMGT v3.1.25 reference sets with S5F mutation patterns.

---

## Docker in depth
<details>
<summary>Step‑by‑step guide</summary>

1. **Pull image**  
   ```bash
   docker pull thomask90/alignair:latest
   ```

2. **Run container**  
   ```bash
   docker run -it --rm \
       -v "/path/to/local/data:/data" \
       thomask90/alignair:latest
   ```

3. **Inside the container, run AlignAIR:**  
   ```bash
   python app.py run \
     --model-checkpoint="/app/pretrained_models/IGH_S5F_576" \
     --save-path="/data" \
     --chain-type=heavy \
     --sequences="/data/test01.csv"
   ```
   Results are written back to your mounted `/data` folder.

4. **For help and all parameters:**
   ```bash
   python app.py run --help
   ```
</details>

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
MIT – see [LICENSE](LICENSE).

---

## Contact
Open an issue or email **thomaskon90@gmail.com**.  
For announcements, visit <https://alignair.ai> or join our Slack.

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

### ✨ Quick Start


```bash
# Pull the latest image
docker pull thomask90/alignair:latest

# Run AlignAIR v2.0 pipeline
docker run -it --rm \
  -v /path/to/local/data:/data \
  -v /path/to/local/downloads:/downloads \
  thomask90/alignair:latest \
  python app.py run \
    --model-checkpoint=/app/pretrained_models/IGH_S5F_576 \
    --genairr-dataconfig=HUMAN_IGH_OGRDB \
    --sequences=/data/sample_HeavyChain_dataset.csv \
    --save-path=/downloads/
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

## What's New in v2.0

AlignAIR v2.0 introduces a revolutionary unified architecture:

### 🔄 Unified Models
- **SingleChainAlignAIR**: Optimized for single receptor type analysis
- **MultiChainAlignAIR**: Native multi-chain support with chain type classification
- **Universal compatibility**: Works with any GenAIRR dataconfig combination

### 🧬 Multi-Chain Analysis
- **Mixed receptor processing**: Analyze IGK + IGL light chains simultaneously
- **Chain type classification**: Automatic receptor type identification
- **Optimized batch processing**: Equal partitioning across chain types

### ⚡ Dynamic GenAIRR Integration
- **Built-in dataconfigs**: `HUMAN_IGH_OGRDB`, `HUMAN_IGK_OGRDB`, `HUMAN_IGL_OGRDB`, `HUMAN_TCRB_IMGT`
- **Custom config support**: Use your own GenAIRR dataconfigs
- **Automatic detection**: Single vs. multi-chain mode based on input

### 📈 Enhanced Performance
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

### Example Commands

**Heavy Chain Analysis:**
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGH_S5F_576 \
  --genairr-dataconfig=HUMAN_IGH_OGRDB \
  --sequences=/data/input/heavy_sequences.csv \
  --save-path=/data/output/heavy_results \
  --v-allele-threshold=0.75 \
  --d-allele-threshold=0.3 \
  --j-allele-threshold=0.8
```

**Light Chain Analysis (Single Chain):**
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGL_S5F_576 \
  --genairr-dataconfig=HUMAN_IGL_OGRDB,HUMAN_IGL_OGRDB \
  --sequences=/data/input/light_sequences.csv \
  --save-path=/data/output/light_results \
  --airr-format \
  --fix-orientation
```

**Multi-Chain Light Chain Analysis (IGK + IGL):**
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/MultiChain_Light_S5F_576 \
  --genairr-dataconfig=HUMAN_IGK_OGRDB,HUMAN_IGL_OGRDB \
  --sequences=/data/input/mixed_light_sequences.csv \
  --save-path=/data/output/multichain_results \
  --airr-format
```

**T-Cell Receptor Beta Chain:**
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/TCRB_Uniform_576 \
  --genairr-dataconfig=HUMAN_TCRB_IMGT \
  --sequences=/data/input/tcr_sequences.csv \
  --save-path=/data/output/tcr_results
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

| DataConfig | Chain Type | Species | Reference | D Gene |
|------------|------------|---------|-----------|--------|
| `HUMAN_IGH_OGRDB` | Heavy Chain | Human | OGRDB v1.5 | ✓ |
| `HUMAN_IGK_OGRDB` | Kappa Light | Human | OGRDB v1.5 | ✗ |
| `HUMAN_IGL_OGRDB` | Lambda Light | Human | OGRDB v1.5 | ✗ |
| `HUMAN_TCRB_IMGT` | TCR Beta | Human | IMGT v3.1.25 | ✓ |
| `HUMAN_IGH_EXTENDED` | Heavy Chain Extended | Human | OGRDB + Custom | ✓ |

### Pre-trained Model Checkpoints

The Docker container ships with optimized models for common use cases:

| Model | Architecture | Supported Configs | Checkpoint Path                             |
|-------|-------------|-------------------|---------------------------------------------|
| **Heavy Chain** | SingleChainAlignAIR | `HUMAN_IGH_OGRDB` | `/app/pretrained_models/IGH_S5F_576`        |
| **Lambda Light** | SingleChainAlignAIR | `HUMAN_IGL_OGRDB` | `/app/pretrained_models/IGL_S5F_576`        |
| **Kappa Light** | SingleChainAlignAIR | `HUMAN_IGK_OGRDB` | `/app/pretrained_models/IGK_S5F_576`        |
| **Multi-Light** | MultiChainAlignAIR | `HUMAN_IGK_OGRDB,HUMAN_IGL_OGRDB` | `/app/pretrained_models/MultiLight_S5F_576` |
| **TCR Beta** | SingleChainAlignAIR | `HUMAN_TCRB_IMGT` | `/app/pretrained_models/TCRB_Uniform_576`   |

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
     --genairr-dataconfig=HUMAN_IGH_OGRDB \
     --sequences="/data/test01.csv" \
     --save-path="/data"
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
This project is licensed under the terms of the [GNU General Public License v3.0 or later (GPLv3+)](LICENSE).

---

## Contact
Open an issue or email **thomaskon90@gmail.com**.  
For announcements, visit <https://alignair.ai> or join our Slack.


<p align="center">
  <img src="https://alignair.ai/_next/static/media/logo_alignair14bw.b74a41a0.svg" width="240" alt="AlignAIR logo">
</p>

<h1 align="center">AlignAIR</h1>
<p align="center">
  <strong>Deep‑learning sequence aligner for immunoglobulin &amp; T‑cell receptor repertoires</strong><br>
  <a href="https://hub.docker.com/r/thomask90/alignair">
    <img alt="Docker pulls" src="https://img.shields.io/docker/pulls/thomask90/alignair">
  </a>
  <a href="https://zenodo.org/record/XXXXXXXX">
    <img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXX.zsvg">
  </a>
  <a href="LICENSE">
    <img alt="MIT" src="https://img.shields.io/badge/license-MIT-green.svg">
  </a>
</p>
 
---

### ✨ Quick Start

```bash
docker run --rm -v "$PWD:/data" thomask90/alignair:latest \
       alignair_predict --mode cli \
       --model_checkpoint /app/pretrained_models/IGH_S5F_576 \
       --sequences /data/my_reads.fasta \
       --save_path /data
```

<details>
<summary>Table of contents</summary>

- [Key features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Docker in depth](#docker-in-depth)
- [Examples](#examples)
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
- **Three interfaces** – CLI, YAML, fully interactive wizard  
- **Drop‑in integration** with AIRR schema & downstream tools (IgBLAST, MiAIRR)

---

## Installation

### Docker (recommended)

```bash
# 1 — pull once
docker pull thomask90/alignair:latest

# 2 — open shell (mount local data to /data)
docker run -it --rm -v /path/to/local/data:/data thomask90/alignair:latest
```

> **Prerequisites:** Nvidia GPU + CUDA 11 recommended (CPU works, slower).

### Local (advanced)

```bash
git clone https://github.com/MuteJester/AlignAIR.git
cd AlignAIR && pip install -r requirements.txt
```

---

## Usage

### CLI

```bash
alignair_predict \
    --model_checkpoint /models/IGH_S5F_576 \
    --sequences my_reads.fasta \
    --chain_type heavy \
    --save_path results/
```

| Flag | Description | Default |
|------|-------------|---------|
| `--batch_size` | inference batch size | `2048` |
| `--v_allele_threshold` | minimum posterior probability | `0.75` |
| `--v_cap` | max V calls returned | `3` |
| *full list:* | `alignair_predict --help` | — |

### YAML

```bash
alignair_predict --mode yaml --config_file configs/heavy.yaml
```

### Interactive wizard

```bash
alignair_predict --mode interactive
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

3. **Choose option 1: CLI mode** from the menu.

4. **Paste CLI arguments**, e.g.  
   ```bash
   --model_checkpoint="/app/pretrained_models/IGH_S5F_576" \
   --save_path="/data" \
   --chain_type=heavy \
   --sequences="/data/test01.fasta"
   ```
   Results are written back to your mounted `/data` folder.
</details>

---

## Examples
See the **`examples/`** folder for Jupyter notebooks:

1. End‑to‑end heavy‑chain pipeline  
2. Benchmark vs. IgBLAST on 10 K reads  
3. Snakemake batch processing  

---

## Data availability
Training & benchmark datasets are archived on Zenodo: `doi:10.5281/zenodo.XXXXXXXX`



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

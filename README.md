![AlignAIR Logo](https://alignair.ai/_next/static/media/logo_alignair14bw.b74a41a0.svg)

# AlignAIR
A Deep Learning-Powered Sequence Alignment Tool for Adaptive Immune Receptors

AlignAIR is a state-of-the-art sequence alignment tool designed to handle the complexities of V(D)J recombination and somatic hypermutation (SHM) in immunoglobulin (Ig) and T cell receptor (TCR) sequences. Leveraging advanced multi-task deep supervised learning, AlignAIR delivers accuracy and efficiency in allele assignment, productivity assessments, and sequence segmentation.

---

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Docker Setup](#docker-setup)
  - [Local Setup](#local-setup)
- [Usage](#usage)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
  - [YAML Configuration](#yaml-configuration)
  - [Interactive Mode](#interactive-mode)
- [Detailed Docker Usage](#detailed-docker-usage)
- [Examples](#examples)
- [Data Availability](#data-availability)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Introduction

AlignAIR is a tool for aligning adaptive immune receptor (AIR) sequences. It addresses the challenges posed by V(D)J recombination and somatic hypermutation (SHM) through a robust multi-task deep learning framework. Whether you're working with immunoglobulin (Ig) or T cell receptor (TCR) sequences, AlignAIR provides high-accuracy allele assignment, sequence segmentation, and productivity assessments—making it an essential tool for AIRR-seq pipelines.

---

## Key Features

- **High Accuracy**: Outperforms existing aligners in allele assignment and sequence segmentation tasks.
- **Scalability**: Efficiently processes millions of sequences with Docker support for consistent environments.
- **Seamless Integration**: Designed to integrate effortlessly with existing AIRR-seq workflows.
- **User-Friendly**: Offers multiple usage modes (CLI, YAML, Interactive) for ease of use.

---

## Installation

### Docker Setup

1. **Pull the Docker image**  
   ```sh
   docker pull thomask90/alignair:latest
   ```
   This command retrieves the latest AlignAIR Docker image from Docker Hub.

2. **Run the Docker container (Interactive Mode)**  
   If you run the container **without** extra arguments, you will be greeted by the AlignAIR menu, which allows you to choose **interactive** or **YAML** modes:
   ```sh
   docker run -it --rm \
       -v /path/to/local/data:/data \
       thomask90/alignair:latest
   ```
   - `-v /path/to/local/data:/data` mounts a local directory (`/path/to/local/data`) to `/data` inside the container.  

3. **Run the Docker container (CLI Mode)**  
   To **bypass** the menu and run AlignAIR directly with your parameters, add `--mode=cli` plus the other required flags:
   ```sh
   docker run -it --rm \
       -v /path/to/local/data:/data \
       thomask90/alignair:latest \
       --mode=cli \
       --model_checkpoint /app/pretrained_models/IGH_S5F_576 \
       --save_path /data/output \
       --chain_type heavy \
       --sequences /data/test.fasta
   ```
   - `--model_checkpoint /app/pretrained_models/IGH_S5F_576` points to a pretrained heavy-chain model included in the container.  
   - `--sequences /data/test.fasta` references your input file (must reside in the mounted `/path/to/local/data` folder).  
   - `--save_path /data/output` is where results will be written inside the container (which maps back to your local folder).

4. **Input File Format**  
   - You can provide FASTA, TSV, or CSV files.  
   - If using CSV/TSV, ensure there's a column called `sequence` which holds your sequences.

5. **Output Files**  
   - Results will appear in the directory you mapped to the container (e.g., `-v /path/to/local/data:/data`) in the location specified by `--save_path` (e.g., `/data/output`).

This setup allows you to either:
- **Explore** via the interactive menu by running with no extra arguments.  
- **Automate** quickly via CLI arguments by specifying `--mode=cli` and any other flags directly.  

### Local Setup

1. **Clone the repository**:
   ```sh
   git clone https://github.com/MuteJester/AlignAIRR.git
   cd AlignAIRR
   ```
2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the model**:
   ```sh
   alignair_predict \
       --mode cli \
       --model_checkpoint /path/to/model_checkpoint.h5 \
       --save_path /path/to/output \
       --chain_type heavy \
       --sequences /path/to/input_sequences.fasta
   ```

---

## Usage

AlignAIR supports three primary modes of operation: **Command Line Interface (CLI)**, **YAML Configuration**, and **Interactive Mode**.

### Command Line Interface (CLI)

```sh
alignair_predict \
    --mode cli \
    --model_checkpoint /path/to/model_checkpoint.h5 \
    --save_path /path/to/output_results \
    --chain_type heavy \
    --sequences /path/to/input_sequences.fasta
```

#### CLI Arguments (Partial List)
| Argument  | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| `--mode`  | Mode of input (`cli`, `yaml`, `interactive`)                                |
| `--model_checkpoint` | Path to saved AlignAIR weights                                   |
| `--save_path`        | Directory to save alignment results                              |
| `--chain_type`       | Type of chain (`heavy`/`light`)                                  |
| `--sequences`        | Path to CSV/TSV/FASTA file containing sequences                  |
| `--batch_size`       | Batch size for model prediction (default: 2048)                  |
| `--v_allele_threshold` | Threshold for V allele prediction (default: 0.75)             |
| `...`                | *See code for additional parameters*                             |

---

### YAML Configuration

For reusable configurations, use a YAML file. For example:

```yaml
mode: yaml
model_checkpoint: /path/to/model_checkpoint.h5
save_path: /path/to/output_results
chain_type: heavy
sequences: /path/to/input_sequences.fasta
batch_size: 2048
v_allele_threshold: 0.75
d_allele_threshold: 0.3
j_allele_threshold: 0.8
v_cap: 3
d_cap: 3
j_cap: 3
translate_to_asc: true
fix_orientation: true
```

Then run:

```sh
alignair_predict --mode yaml --config_file /path/to/config.yaml
```

---

### Interactive Mode

If you prefer a guided setup, run:

```sh
alignair_predict --mode interactive
```

You will be prompted to provide the model checkpoint path, save path, chain type, sequence file path, etc.

---

## Detailed Docker Usage

If you want a step-by-step guide for running AlignAIR **interactively inside Docker** and selecting the CLI mode:

1. **Pull the Docker image** (only needed once):  
   ```sh
   docker pull thomask90/alignair:latest
   ```
2. **Run the container in interactive mode**:
   ```sh
   docker run -it --rm \
       -v "/path/to/local/data:/data" \
       thomask90/alignair:latest
   ```
   - `-it` opens an interactive terminal session.
   - `--rm` removes the container when it exits.
   - `-v "/path/to/local/data:/data"` mounts your local directory to `/data` in the container.

3. **Select CLI Mode**  
   Inside the container, you’ll see a menu:
   ```
   1. Run Model in CLI Mode
   2. Run Model with YAML Configuration
   3. Run Model in Interactive Mode
   4. Exit
   ```
   Type `1` and press **Enter**.

4. **Enter Your Model Parameters**  
   You’ll be prompted to paste CLI arguments. For example:
   ```bash
   --model_checkpoint="/app/pretrained_models/IGH_S5F_576" \
   --save_path="/data/" \
   --chain_type=heavy \
   --sequences="/data/test01.fasta"
   ```
   - **`/app/pretrained_models/IGH_S5F_576`** is a built-in pretrained heavy-chain model.
   - **`/app/pretrained_models/IGL_S5F_576`** is a built-in pretrained light-chain model.
   - **`/data/`** is your mounted local directory inside the container.
   - **`test01.fasta`** is your input file, which must be located in the local folder you mounted.

That’s it! Your results will be saved to `/data/` inside the container, which corresponds to your local directory outside the container.

---

## Examples

### Basic CLI Usage
```sh
alignair_predict \
    --mode cli \
    --model_checkpoint /path/to/model_checkpoint.h5 \
    --save_path /path/to/output_results \
    --chain_type heavy \
    --sequences /path/to/input_sequences.fasta
```

### YAML Configuration Usage
```sh
alignair_predict --mode yaml --config_file /path/to/config.yaml
```

### Interactive Mode Usage
```sh
alignair_predict --mode interactive
```

---

## Data Availability

The datasets used for training and evaluating AlignAIR are available on Zenodo:

- [Training Datasets](https://zenodo.org/record/training_datasets)
- [Evaluation Datasets](https://zenodo.org/record/evaluation_datasets)

---

## Contributing

We welcome contributions from the community! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

---

## License

AlignAIR is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, feedback, or support, please reach out to the project maintainer at [thomaskon90@gmail.com](mailto:thomaskon90@gmail.com).

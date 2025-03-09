# AlignAIR

![AlignAIR Logo](https://alignair.ai/_next/static/media/logo_alignair14bw.b74a41a0.svg) <!-- Add your project logo here if you have one -->

**AlignAIR**: A Deep Learning-Powered Sequence Alignment Tool for Adaptive Immune Receptors

AlignAIR is a state-of-the-art sequence alignment tool designed to handle the complexities of V(D)J recombination and somatic hypermutation (SHM) in immunoglobulin (Ig) and T cell receptor (TCR) sequences. Leveraging advanced multi-task deep supervised learning, AlignAIR delivers unparalleled accuracy and efficiency in allele assignment, productivity assessments, and sequence segmentation.

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
- [Examples](#examples)
- [Data Availability](#data-availability)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Introduction

AlignAIR is a tool for aligning adaptive immune receptor (AIR) sequences. It addresses the challenges posed by V(D)J recombination and somatic hypermutation (SHM) through a robust multi-task deep learning framework. Whether you're working with immunoglobulin (Ig) or T cell receptor (TCR) sequences, AlignAIR provides high-accuracy allele assignment, sequence segmentation, and productivity assessments, making it an essential tool for AIRR-seq pipelines.

---

## Key Features

- **High Accuracy**: Outperforms existing aligners in allele assignment and sequence segmentation tasks.
- **Scalability**: Efficiently processes millions of sequences with Docker support for consistent environments.
- **Seamless Integration**: Designed to integrate effortlessly with existing AIRR-seq workflows.
- **User-Friendly**: Offers a web interface and multiple usage modes (CLI, YAML, Interactive) for ease of use.

---

## Installation

### Docker Setup

1. **Pull the Docker image**:
   ```sh
   docker pull alignair/alignair:latest
   ```

2. **Run the Docker container**:
   ```sh
   docker run -v /path/to/data:/data alignair/alignair:latest --mode cli --model_checkpoint /data/model_checkpoint.h5 --save_path /data/output_results.txt --chain_type heavy --sequences /data/input_sequences.csv
   ```

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
   python alignair.py --mode cli --model_checkpoint /path/to/model_checkpoint.h5 --save_path /path/to/output_results.txt --chain_type heavy --sequences /path/to/input_sequences.csv
   ```

---

## Usage

AlignAIR supports three modes of operation: **Command Line Interface (CLI)**, **YAML Configuration**, and **Interactive Mode**.

### Command Line Interface (CLI)

Run AlignAIR directly from the command line by specifying input files, output directories, and other parameters.

```sh
python alignair.py --mode cli --model_checkpoint /path/to/model_checkpoint.h5 --save_path /path/to/output_results.txt --chain_type heavy --sequences /path/to/input_sequences.csv --batch_size 2048 --v_allele_threshold 0.75 --d_allele_threshold 0.3 --j_allele_threshold 0.8
```

#### CLI Arguments
| Argument | Description |
|----------|-------------|
| `--mode` | Mode of input (`cli`, `yaml`, `interactive`) |
| `--config_file` | Path to YAML configuration file (YAML mode only) |
| `--model_checkpoint` | Path to saved AlignAIR weights |
| `--save_path` | Directory to save alignment results |
| `--chain_type` | Type of chain (`heavy`, `light`) |
| `--sequences` | Path to CSV/TSV file containing sequences |
| `--lambda_data_config` | Path to lambda chain data config (default: `D`) |
| `--kappa_data_config` | Path to kappa chain data config (default: `D`) |
| `--heavy_data_config` | Path to heavy chain data config (default: `D`) |
| `--max_input_size` | Maximum model input size (default: 576) |
| `--batch_size` | Batch size for model prediction (default: 2048) |
| `--v_allele_threshold` | Threshold for V allele prediction (default: 0.75) |
| `--d_allele_threshold` | Threshold for D allele prediction (default: 0.3) |
| `--j_allele_threshold` | Threshold for J allele prediction (default: 0.8) |
| `--v_cap` | Cap for V allele calls (default: 3) |
| `--d_cap` | Cap for D allele calls (default: 3) |
| `--j_cap` | Cap for J allele calls (default: 3) |
| `--translate_to_asc` | Translate names back to ASCs from IMGT |
| `--fix_orientation` | Fix DNA orientation (default: `True`) |
| `--custom_orientation_pipeline_path` | Path to a custom orientation model |

---

### YAML Configuration

For reusable configurations, use a YAML file to define your settings.

#### Example YAML Configuration
```yaml
mode: yaml
model_checkpoint: /path/to/model_checkpoint.h5
save_path: /path/to/output_results.txt
chain_type: heavy
sequences: /path/to/input_sequences.csv
lambda_data_config: D
kappa_data_config: D
heavy_data_config: D
max_input_size: 576
batch_size: 2048
v_allele_threshold: 0.75
d_allele_threshold: 0.3
j_allele_threshold: 0.8
v_cap: 3
d_cap: 3
j_cap: 3
translate_to_asc: true
fix_orientation: true
custom_orientation_pipeline_path: /path/to/custom_orientation_model
```

#### Run with YAML
```sh
python alignair.py --mode yaml --config_file /path/to/config.yaml
```

---

### Interactive Mode

AlignAIR's interactive mode guides you through setting up your configuration step-by-step.

#### Run Interactive Mode
```sh
python alignair.py --mode interactive
```

You will be prompted to provide:
- Path to saved AlignAIR weights
- Save path for alignment results
- Chain type (`heavy` or `light`)
- Path to CSV/TSV file with sequences
- Data configurations for lambda, kappa, and heavy chains
- Maximum input size
- Batch size
- Thresholds for V, D, and J allele predictions
- Caps for V, D, and J allele calls
- Options for translating to ASCs and fixing orientation
- Path to a custom orientation model

---

## Examples

### Basic CLI Usage
```sh
python alignair.py --mode cli --model_checkpoint /path/to/model_checkpoint.h5 --save_path /path/to/output_results.txt --chain_type heavy --sequences /path/to/input_sequences.csv
```

### YAML Configuration Usage
```sh
python alignair.py --mode yaml --config_file /path/to/config.yaml
```

### Interactive Mode Usage
```sh
python alignair.py --mode interactive
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
```


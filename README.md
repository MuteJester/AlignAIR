# AlignAIR

![AlignAIR Logo](https://alignair.ai/logo.png) <!-- Add your project logo here if you have one -->

AlignAIR: Sequence Alignment of Adaptive Immune Receptors Enhanced by Multi-Task Deep Supervised Learning

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Docker Setup](#docker-setup)
  - [Local Setup](#local-setup)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [YAML Configuration](#yaml-configuration)
  - [Interactive Mode](#interactive-mode)
- [Examples](#examples)
- [Data Availability](#data-availability)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

AlignAIR is a novel deep learning-based aligner designed to handle the complexities of V(D)J recombination and somatic hypermutation (SHM) in immunoglobulin (Ig) and T cell receptor (TCR) sequences. It leverages advanced simulation approaches and a multi-task learning framework to achieve high accuracy in allele assignment, productivity assessments, and sequence segmentation.

## Features

- **High Accuracy**: Surpasses state-of-the-art aligners in allele assignment accuracy and sequence segmentation.
- **Speed**: Efficient processing of millions of sequences with Docker support.
- **Integration**: Designed for seamless integration with existing AIRR-seq pipelines.
- **User-Friendly**: Web interface for easy usage and Docker image for consistent environment setup.

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

2. **Install the dependencies**:

    ```sh
    pip install -r requirements.txt
    ```

3. **Run the model**:

    ```sh
    python alignair.py --mode cli --model_checkpoint /path/to/model_checkpoint.h5 --save_path /path/to/output_results.txt --chain_type heavy --sequences /path/to/input_sequences.csv
    ```

## Usage

AlignAIR offers three modes of operation: CLI, YAML, and Interactive. 

### Command Line Interface (CLI)

To use AlignAIR via the command line, specify the input file, output directory, and other parameters as needed.

```sh
python alignair.py --mode cli --model_checkpoint /path/to/model_checkpoint.h5 --save_path /path/to/output_results.txt --chain_type heavy --sequences /path/to/input_sequences.csv --batch_size 2048 --v_allele_threshold 0.75 --d_allele_threshold 0.3 --j_allele_threshold 0.8
```

**CLI Arguments**:
- `--mode`: Mode of input (`cli`, `yaml`, `interactive`)
- `--config_file`: Path to YAML configuration file (only for YAML mode)
- `--model_checkpoint`: Path to saved AlignAIR weights
- `--save_path`: Directory to save the alignment results
- `--chain_type`: Type of chain (`heavy`, `light`)
- `--sequences`: Path to CSV/TSV file with sequences
- `--lambda_data_config`: Path to lambda chain data config (default: `D`)
- `--kappa_data_config`: Path to kappa chain data config (default: `D`)
- `--heavy_data_config`: Path to heavy chain data config (default: `D`)
- `--max_input_size`: Maximum model input size (default: 576)
- `--batch_size`: Batch size for model prediction (default: 2048)
- `--v_allele_threshold`: Threshold for V allele prediction (default: 0.75)
- `--d_allele_threshold`: Threshold for D allele prediction (default: 0.3)
- `--j_allele_threshold`: Threshold for J allele prediction (default: 0.8)
- `--v_cap`: Cap for V allele calls (default: 3)
- `--d_cap`: Cap for D allele calls (default: 3)
- `--j_cap`: Cap for J allele calls (default: 3)
- `--translate_to_asc`: Translate names back to ASCs from IMGT
- `--fix_orientation`: Fix DNA orientation (default: True)
- `--custom_orientation_pipeline_path`: Path to a custom orientation model

### YAML Configuration

You can also run AlignAIR using a YAML configuration file. This is useful for saving and reusing configurations.

**Example YAML Configuration**:
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

**Run with YAML**:
```sh
python alignair.py --mode yaml --config_file /path/to/config.yaml
```

### Interactive Mode

The interactive mode guides you through a series of questions to set up your configuration.

**Run Interactive Mode**:
```sh
python alignair.py --mode interactive
```

You will be prompted to provide:
- Path to saved AlignAIR weights
- Save path for the alignment results
- Chain type (heavy/light)
- Path to CSV/TSV file with sequences
- Data configurations for lambda, kappa, and heavy chains
- Maximum input size
- Batch size
- Thresholds for V, D, and J allele predictions
- Caps for V, D, and J allele calls
- Options for translating to ASCs and fixing orientation
- Path to a custom orientation model

## Examples

Here are some example commands and use cases for AlignAIR.

**Basic CLI Usage**:
```sh
python alignair.py --mode cli --model_checkpoint /path/to/model_checkpoint.h5 --save_path /path/to/output_results.txt --chain_type heavy --sequences /path/to/input_sequences.csv
```

**YAML Configuration Usage**:
```sh
python alignair.py --mode yaml --config_file /path/to/config.yaml
```

**Interactive Mode Usage**:
```sh
python alignair.py --mode interactive
```

## Data Availability

The datasets used for training and evaluation are available on Zenodo:

- [Training datasets](https://zenodo.org/record/training_datasets)
- [Evaluation datasets](https://zenodo.org/record/evaluation_datasets)

## Contributing

We welcome contributions to AlignAIR! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for more details on how to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or support, please contact the project maintainer at [thomaskon90@gmail.com](mailto:thomaskon90@gmail.com).

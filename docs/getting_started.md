# Getting Started with AlignAIR

**AlignAIR** is a multi-task deep learning based immunoglobulin (Ig) sequence alignment suite.
It combines a sophisticated pipeline with extensive preprocessing steps leading up to the model ensuring optimal and
correct input adjustment, including detection and correction of sequence orientation, handling of long read sequences.
After the model there are several post processing steps that ensure a uniform format following the AIRR schema ensuring compatibility with other sequence processing and analysis algorithms.

## Tutorials and Examples

To start aligning sequences you can either train your own custom model finetuned to your specific reference/species or use our pretrained models. Below are examples for different use cases and useful tutorials:

- **[Training and Using AlignAIR in Jupyter IDE](tutorials/AlignAIR_On_Jupyter_Notebooks.ipynb)**: Browse a collection of example models that showcase different capabilities of GenAIRR, providing insights into how to build and extend the framework for specific needs.

### Community and Support

- **[AlignAIR Issues](https://github.com/MuteJester/AlignAIR/issues)**: Report bugs or request features.

## Docker Installation and Usage

You can run AlignAIR using Docker. First, build the Docker image:

```bash
docker build -t alignair_cli .
```

Then run the container with mounted volume:

```bash
docker run -it -v "${PWD}:/local_server" alignair_interactive_cli
```

## Quick Start with Docker CLI

Here's an example of running AlignAIR in Docker with specific arguments:

```bash
docker run -it -v "${PWD}:/local_server" alignair_interactive_cli \
    --model_checkpoint=/app/tests/AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2 \
    --save_path=/app/tests/ \
    --chain_type=heavy \
    --sequences=/app/tests/sample_HeavyChain_dataset.csv \
    --batch_size=32 \
    --translate_to_asc
```

## Features

- Fast and accurate immunoglobulin sequence alignment
- Pre-trained models for common species and chain types
- Support for custom model training
- Extensive preprocessing pipeline
- AIRR-compliant output format
- Handles both short and long read sequences
- Automatic sequence orientation correction

## Requirements

- Docker
- 8GB RAM minimum
- 6GB disk space

For detailed system requirements, please check the documentation in the repository.

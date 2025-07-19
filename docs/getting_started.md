# Getting Started with AlignAIR v2.0

**AlignAIR v2.0** introduces a revolutionary unified architecture for immunoglobulin (Ig) and T-cell receptor (TCR) sequence alignment. The new system features dynamic GenAIRR integration, universal model architectures, and native multi-chain support, making it faster and more flexible than ever.

## What's New in v2.0

### Unified Architecture
- **SingleChainAlignAIR**: Optimized for single receptor type analysis
- **MultiChainAlignAIR**: Native multi-chain support with automatic chain type classification
- **Universal Datasets**: SingleChainDataset and MultiChainDataset classes that work with any GenAIRR dataconfig

### Dynamic GenAIRR Integration
- Built-in dataconfigs: `HUMAN_IGH_OGRDB`, `HUMAN_IGK_OGRDB`, `HUMAN_IGL_OGRDB`, `HUMAN_TCRB_IMGT`
- Custom dataconfig support via pickle files
- Automatic chain type detection and model selection

### Enhanced Multi-Chain Capabilities
- Mixed light chain analysis (IGK + IGL)
- Simultaneous processing of different receptor types
- Chain type classification as model output
- Optimized batch processing for multi-chain scenarios

## Basic Usage Examples

### Single Heavy Chain Analysis
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGH_S5F_576 \
  --genairr-dataconfig=HUMAN_IGH_OGRDB \
  --sequences=heavy_sequences.csv \
  --save-path=results/
```

### Single Light Chain Analysis (Lambda)
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/IGL_S5F_576 \
  --genairr-dataconfig=HUMAN_IGL_OGRDB \
  --sequences=lambda_sequences.csv \
  --save-path=results/
```

### Multi-Chain Light Chain Analysis
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/MultiLight_S5F_576 \
  --genairr-dataconfig=HUMAN_IGK_OGRDB,HUMAN_IGL_OGRDB \
  --sequences=mixed_light_sequences.csv \
  --save-path=results/
```

### TCR Beta Chain Analysis
```bash
python app.py run \
  --model-checkpoint=/app/pretrained_models/TCRB_S5F_576 \
  --genairr-dataconfig=HUMAN_TCRB_IMGT \
  --sequences=tcr_sequences.csv \
  --save-path=results/
```

## Tutorials and Examples

To start aligning sequences you can either train your own custom model finetuned to your specific reference/species or use our pretrained models. Below are examples for different use cases and useful tutorials:

- **[Training and Using AlignAIR v2.0 in Jupyter IDE](tutorials/AlignAIR_On_Jupyter_Notebooks.ipynb)**: Comprehensive guide to the new unified architecture with examples of both single-chain and multi-chain workflows.

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

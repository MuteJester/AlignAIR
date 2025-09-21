# AlignAIR

[AlignAIR](https://github.com/MuteJester/AlignAIR) is an AGPL-3 licensed sequence alignment tool specifically designed for Adaptive Immune Receptor Repertoire (AIRR) sequences. AlignAIR v2.0 features a unified architecture that dynamically supports both single-chain and multi-chain analysis with seamless GenAIRR integration.

## Overview

AlignAIR v2.0 represents a major architectural advancement, combining powerful sequence alignment algorithms with a unified, modular interface. The new system features:

- **Unified Model Architecture**: SingleChainAlignAIR and MultiChainAlignAIR models that automatically adapt to different receptor types
- **Dynamic GenAIRR Integration**: Built-in support for GenAIRR dataconfigs with automatic chain type detection
- **Multi-Chain Analysis**: Native support for analyzing mixed receptor populations (e.g., IGK + IGL light chains)
- **Flexible Data Handling**: Universal dataset classes that work with any combination of chain types

## Features

- **Unified Multi-Chain Framework**: AlignAIR v2.0 introduces a revolutionary unified architecture that seamlessly handles single-chain and multi-chain scenarios
- **Dynamic GenAIRR Integration**: Built-in support for GenAIRR dataconfigs with automatic chain type detection and configuration
- **Universal Model Architecture**: SingleChainAlignAIR and MultiChainAlignAIR models that adapt to any receptor combination
- **Enhanced Multi-Task Learning**: Joint optimization of V/D/J segmentation, allele calling, chain type classification, and sequence analysis
- **Scalable Performance**: Optimized for large-scale AIRR data with efficient batch processing and GPU acceleration
- **Comprehensive Chain Support**: Native support for IGH, IGK, IGL, TCRB, and custom receptor types

## Installation Options

To install the latest stable release of AlignAIR, use:

```bash
pip install AlignAIR
```

For installation from the GitHub repository for the latest development version:

```bash
git clone https://github.com/MuteJester/AlignAIR.git
cd AlignAIR
pip install .
```

## Quick Start Guide

1. **Input Preparation**: Ensure your input data is in a compatible format (e.g., FASTA or CSV with sequences).
2. **Choose Configuration**: Select appropriate GenAIRR dataconfig(s) for your receptor type(s):
   - Single chain: `--genairr-dataconfig=HUMAN_IGH_OGRDB`
   - Multi-chain: `--genairr-dataconfig=HUMAN_IGK_OGRDB,HUMAN_IGL_OGRDB`
3. **Running AlignAIR**: Use the unified CLI interface:
    ```bash
    python app.py run --model-dir=model_path \
                     --genairr-dataconfig=HUMAN_IGH_OGRDB \
                     --sequences=my_sequences.csv \
                     --save-path=results/
    ```
4. **Results**: AlignAIR automatically detects single vs. multi-chain scenarios and adapts accordingly.

## Docker Support

AlignAIR provides a Docker image to ensure a consistent runtime environment. To use AlignAIR with Docker:

1. **Pull the Docker Image**:
    ```bash
    docker pull thomask90/alignair:latest
    ```

2. **Run the Container (entrypoint style)**:
        ```bash
        docker run -it --rm \
            -v $(pwd):/data \
            -v $(pwd)/results:/downloads \
            thomask90/alignair:latest run \
            --model-dir=/app/pretrained_models/IGH_S5F_576 \
            --genairr-dataconfig=HUMAN_IGH_OGRDB \
            --sequences=/data/my_sequences.fasta \
            --save-path=/downloads/
        ```

## Documentation Resources

- [Getting Started Guide](link-to-guide) – Learn how to set up and start using AlignAIR.
- [Configuration Reference](link-to-config-doc) – Detailed explanations of all customizable parameters.
- [CLI Commands](link-to-cli-doc) – A comprehensive list of commands and options.
- [Examples and Tutorials](link-to-examples) – Hands-on examples to help users familiarize themselves with AlignAIR's capabilities.

## Development and Contributions

AlignAIR is an open-source project. We welcome contributions from the community:

- [GitHub Repository](https://github.com/MuteJester/AlignAIR)
- [Contributing Guide](https://github.com/MuteJester/AlignAIR/blob/main/CONTRIBUTING.md)
- [Issue Tracker](https://github.com/MuteJester/AlignAIR/issues) – Report bugs or suggest new features.

## Publications and References

The detailed methodology and performance benchmarks are discussed in the main manuscript and supplementary documentation [here](link-to-publication).

[contributors guide]: https://github.com/MuteJester/AlignAIR/blob/main/CONTRIBUTING.md
[github repository]: https://github.com/MuteJester/AlignAIR
[issue tracker]: https://github.com/MuteJester/AlignAIR/issues
[examples and tutorials]: https://github.com/MuteJester/AlignAIR/tree/main/docs/tutorials

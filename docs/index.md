# AlignAIR

[AlignAIR](https://github.com/MuteJester/AlignAIR) is an AGPL-3 licensed sequence alignment tool specifically designed for Adaptive Immune Receptor Repertoire (AIRR) sequences. AlignAIR is a comprehensive suite developed for researchers and practitioners who need precise and efficient analysis and alignment of Ig sequences.

## Overview

AlignAIR combines advanced sequence alignment algorithms with a user-friendly, modular interface that allows for extensive customization and integration. The suite is designed to handle large-scale AIRR data, supporting various sequence types and facilitating comparative studies and benchmarking of alignment methods.

## Features

- **Multi-Task Deep Learning Framework**: AlignAIR is built with a deep learning model capable of handling complex alignments while preserving accuracy.
- **Built-in Support for Common Datasets**: The package includes configurations for AIRR datasets, enabling easy setup and use.
- **Robust Ambiguity Resolution**: Provides reliable allele calls and ensures precise meta-information for each sequence.

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
2. **Running AlignAIR**: Use the CLI or Python script to initiate the alignment process:
    ```bash
    alignair --input my_sequences.fasta --output results/
    ```
3. **Customization**: Modify configuration files for specific alignment needs or custom modules.

## Docker Support

AlignAIR provides a Docker image to ensure a consistent runtime environment. To use AlignAIR with Docker:

1. **Pull the Docker Image**:
    ```bash
    docker pull mutejester/alignair:latest
    ```

2. **Run the Container**:
    ```bash
    docker run -v $(pwd):/data mutejester/alignair:latest --input /data/my_sequences.fasta --output /data/results/
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

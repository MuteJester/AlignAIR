# Security Policy

## Reporting a vulnerability

Please report security issues privately to **thomaskon90@gmail.com** rather than opening a public
issue. Include a description, reproduction steps, and the affected version. We aim to acknowledge
reports within a few business days.

## Scope

AlignAIR distributes models as `.alignair` files. The sections used for **inference** — the config
(JSON), the weights (safetensors), and the embedded germline reference (`reference_json`) — are
portable and load **without executing any pickle**, and the embedded reference's allele-order and
FASTA fingerprints are hash-verified on every load. Models published for inference contain zero
pickle sections.

A `.alignair` file may additionally carry *resumable-checkpoint* sections (`dataconfig`,
`train_state`) that use Python pickle / `torch.load` and therefore execute code on load. Only load
checkpoints — or any model file — from sources you trust. Models downloaded from the official catalog
are fetched over HTTPS from the Hugging Face Hub and verified against the catalog's SHA-256.

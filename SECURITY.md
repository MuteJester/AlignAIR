# Security Policy

## Reporting a vulnerability

Please report security issues privately to **thomaskon90@gmail.com** rather than opening a public
issue. Include a description, reproduction steps, and the affected version. We aim to acknowledge
reports within a few business days.

## Scope

AlignAIR loads model **bundles** that include a SHA-256 `fingerprint.txt` verified on load
(tamper/corruption detection). Only load bundles from sources you trust — a model file is code-
adjacent data (`torch.load`). Downloaded models from the official catalog are fetched over HTTPS
from the Hugging Face Hub.

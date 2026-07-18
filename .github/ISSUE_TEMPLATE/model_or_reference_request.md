---
name: Model or reference request
about: Request a pretrained model, or report a problem training/aligning with a reference
title: "[model/reference] "
labels: model-request
---

**What do you need?**
- [ ] A pretrained model for a species/locus (e.g. mouse IGK)
- [ ] Help training on my own reference (FASTA / OGRDB set)
- [ ] A reference-parsing or training failure

**Species / locus / chain type**
<!-- e.g. human IGH (BCR_HEAVY), mouse TCRB (TCR_BETA) -->

**Reference source**
<!-- OGRDB set + version, IMGT, a custom FASTA, or a GenAIRR built-in dataconfig name -->

**If a training/alignment failure, paste:**
- the exact `alignair train ...` / `alignair predict ...` command
- the error output
- your `alignair doctor` output

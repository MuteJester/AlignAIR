# Getting started

A 5‑minute walk‑through of AlignAIR: install, check your environment, and align reads — including
against your own donor/novel genotype.

## 1. Install

```bash
pip install "AlignAIR[cli]"            # core + CLI
pip install "AlignAIR[cli,reader]"     # optional: parasail (faster, sharper V calling)
```

Check the environment (Python, PyTorch + CUDA, GenAIRR, optional parasail):

```bash
alignair doctor
```

GPU is auto‑detected; pass `--device cpu` or `--device cuda` to `alignair predict` to force one.

## 2. Get a model

`alignair predict` needs a model — either a **bundle** directory or a raw `.pt` checkpoint. Options:

- Use a pretrained bundle (published on the project's model hub; a `alignair model download`
  command is on the [roadmap](architecture/adoption_roadmap.md)).
- **Train your own** for any reference or species:
  ```bash
  # any of GenAIRR's ~90 built-in references
  alignair train --reference HUMAN_IGH_OGRDB -o runs/my_igh --preset desktop
  # or your own germline FASTAs (custom/novel reference)
  alignair train --v-fasta v.fa --d-fasta d.fa --j-fasta j.fa --chain-type BCR_HEAVY \
    -o runs/custom --preset desktop --allow-curatable
  ```
  This writes `runs/.../bundle/` (self-contained — custom references are embedded), plus a
  `model_card.md` and `validation_report.json`. Use it with `--model runs/.../bundle`.
- Package a raw checkpoint into a versioned bundle: `alignair bundle --model ckpt.pt -o my_bundle/`.

## 3. Align reads

Input may be FASTA, FASTQ, CSV/TSV, or TXT (optionally `.gz`):

```bash
alignair predict examples/reads.fasta -o out.tsv --model <bundle_or_checkpoint>
```

The output is an AIRR rearrangement TSV with `v_call`/`d_call`/`j_call`, per‑gene sequence and
germline coordinates, `junction`/`junction_aa`, `productive`, `rev_comp`, and calibrated
uncertainty columns (`*_call_set`, `*_call_level`, `*_set_confidence`).

## 4. Use your own reference (dynamic genotype)

The reference is an **input**. Supply a genotype as YAML or FASTA — it can contain **fewer alleles**
than the trained reference and/or **novel alleles** the model has never seen:

```bash
# YAML: top-level v/d/j, each {allele_name: dna_sequence}
alignair predict examples/reads.fasta -o out.tsv \
  --model <bundle_or_checkpoint> --genotype examples/donor_genotype.yaml

# FASTA: >allele_name headers (gene type inferred from the AIRR/IMGT name)
alignair predict examples/reads.fasta -o out.tsv \
  --model <bundle_or_checkpoint> --genotype donor.fasta
```

No retraining is required — the model conditions on exactly the alleles you provide, and every call
is guaranteed to be within that genotype.

## 5. Common options

| Flag | Meaning |
| --- | --- |
| `--genotype FILE` | YAML/FASTA reference for this run (subset and/or novel alleles) |
| `--calibration FILE` | allele‑set calibration JSON (overrides a bundled one) |
| `--v-reader parasail` | use the fast classical V reader (needs `AlignAIR[reader]`) |
| `--batch N` | batch size (default 64) |
| `--device cuda|cpu` | force a device (auto if unset) |
| `--quiet` | suppress progress output |

## Next steps

- [DNAlignAIR design & benchmarks](dnalignair.md) — architecture and head‑to‑head results.
- [Adoption roadmap](architecture/adoption_roadmap.md) — model zoo, `alignair train`, packaging.

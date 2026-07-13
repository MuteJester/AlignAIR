# AlignAIR model contract (v1 — fixed-reference classifier)

**Status:** frozen for the v2.x release series. **Architecture contract:** `fixed-reference` (v1).
This is versioned **separately** from the `.alignair` container format version — a future
runtime-reference architecture would be a new contract, not a container-format bump.

## What the model is

An AlignAIR model is a **fixed-reference classifier**. Its V/D/J classification heads have a fixed
output dimension whose indices are tied, in order, to the germline allele catalog the model was trained
on. That catalog travels with the model (embedded, hash-verified `reference_json`) and **is** the set of
alleles the model can call.

## What is supported

- **Prediction over the trained catalog.** Every call is one of the embedded reference's alleles.
- **Donor genotype subsetting.** A JSON/YAML genotype (`{gene: [allele names]}`) that is a **subset**
  of the trained reference constrains calls to that donor's alleles (`predict --genotype`, methods
  `mask`/`softmax`/`renormalize`). Calls are hard-restricted to the allowed set (P0-5); partial genes
  are fine; the confidence and accuracy interactions are documented in the genotype study.
- **Multi-chain unions.** A model trained on several loci carries a per-locus schema; each read is
  attributed to a locus and can only call that locus's alleles (P0-6). Cross-locus calls are impossible.

## What is NOT supported (requires retraining / fine-tuning)

- **Novel alleles.** An allele absent from the trained catalog cannot be called. Supplying one (e.g. in
  a genotype or a caller-provided reference) fails immediately with
  `alignair.genotype.NovelAlleleUnsupportedError` — it is never silently dropped or mis-indexed.
- **Adding alleles / species at inference.** Extend the catalog by training or fine-tuning a new model
  against the new reference; that produces a new fixed head with a new (versioned) catalog.

## Guarantees enforced in code

- `load_model` rejects a caller-supplied reference whose ordered allele identity does not exactly match
  the embedded head (`_assert_reference_matches`) — no output-column mislabeling.
- Genotype constraints validate a non-empty allowed set per constrained gene before inference, and
  reject novel alleles (`NovelAlleleUnsupportedError`).
- The embedded reference's allele-order and FASTA hashes are verified on load.

The acceptance tests for this contract live in `tests/alignair/test_fixed_reference_contract.py`.

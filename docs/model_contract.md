# Model contract

An AlignAIR model is a **fixed-reference classifier**. Understanding this one property explains what a
model can and cannot do at inference.

## What a model is

The V/D/J classification heads have a fixed output size whose indices are tied, in order, to the
germline allele catalog the model was trained on. That catalog travels **inside** the model file
(embedded and hash-verified) and **is** the exact set of alleles the model can call.

## Supported at inference

- **Prediction over the trained catalog.** Every call is one of the embedded reference's alleles.
- **Donor-genotype subsetting.** A YAML/JSON genotype (`{gene: [allele names]}`) that is a **subset** of
  the trained reference constrains calls to that donor's alleles (`alignair predict --genotype`, with
  `--genotype-method mask`/`softmax`/`renormalize`). Calls are hard-restricted to the allowed set;
  partial genes are fine. See [Donor genotype](getting_started.md#4-constrain-to-a-donor-genotype).
- **Multi-locus models.** A model trained on several loci carries a per-locus schema; each read is
  attributed to a locus and can only call that locus's alleles. Cross-locus calls are impossible.

## Not supported — requires training or fine-tuning

- **Novel alleles.** An allele absent from the trained catalog cannot be called. Supplying one (in a
  genotype or a caller-provided reference) fails immediately with a clear error — it is never silently
  dropped or mis-indexed.
- **Adding alleles / a new species / a new locus.** This changes the model's allele universe, so it
  requires **training or fine-tuning** a new model against the new reference (see
  [Train your own](getting_started.md#2-get-a-model)). That produces a new model with a new, versioned
  catalog.

## Safety guarantees enforced on load

- A caller-supplied reference whose ordered allele identity does not exactly match the embedded head is
  rejected — so output columns can never be silently mislabelled.
- The embedded reference's allele-order and FASTA fingerprints are verified every time a model loads.
- Genotype constraints validate a non-empty allowed set per constrained gene before inference and reject
  novel alleles.

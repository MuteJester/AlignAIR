# Novel alleles & allele subsets (dynamic genotype)

AlignAIR's reference is a **runtime input**, so a donor genotype can contain **fewer**
alleles than the trained reference and/or **novel** alleles the model has never seen — no
retraining. [`donor_with_novel.yaml`](donor_with_novel.yaml) is a small human-IGH genotype
(a subset of the OGRDB reference) plus one novel V allele (`IGHVF1-G1*NOVEL01`, a real V
with three point differences).

```bash
alignair predict --input reads.fasta --out out.tsv --model my_model/bundle \
    --genotype examples/novel_allele/donor_with_novel.yaml
```

Reads from the novel allele are aligned against it directly — the model conditions on the
sequences you provide rather than memorising a fixed allele set. A YAML/FASTA genotype
carries sequences only; junction derivation needs the conserved anchors, so junctions are
omitted for genotype-supplied alleles (see [Known failure modes](../../docs/known_failure_modes.md)).

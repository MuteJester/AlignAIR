# Donor genotype subsets

A donor genotype restricts prediction to a **subset of the model's trained reference** — the alleles
known to be present in that donor. This sharpens accuracy on ambiguous reads. It is applied at predict
time with no retraining:

```bash
alignair predict --input reads.fasta --out out.tsv --model my_model/bundle/model.alignair \
    --genotype examples/novel_allele/donor.yaml
```

The genotype names alleles that must be a subset of the model's reference. **Novel alleles** (alleles
the model was not trained on), a **new species**, or a **changed allele universe** are not callable by a
fixed-reference model — supplying one is rejected with a clear error. To support them, train or
fine-tune a compatible model against the new reference (see [`../custom_reference/`](../custom_reference/)).

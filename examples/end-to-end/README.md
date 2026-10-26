# Real-data walkthrough

The [worked example](https://mutejester.github.io/AlignAIR/#/docs/worked-example) contains
the dataset details, download commands, checksums, recorded results, and interpretation.
This folder supplies [merge_pairs.py](merge_pairs.py), its demonstration read merger.

After downloading the two subsets into `aa_example/` as described in the guide, run
from the repository root:

```bash
python3 examples/end-to-end/merge_pairs.py aa_example/R1.sub.fastq aa_example/R2.sub.fastq aa_example/merged.fastq
alignair predict --input aa_example/merged.fastq --out aa_example/briney_merged.tsv \
  --model alignair-igh-human@1.0.0
alignair validate-airr aa_example/briney_merged.tsv
alignair analyze aa_example/briney_merged.tsv --format text
```

The merger drops pairs without an accepted overlap and writes placeholder quality scores.
It supports the reproducible tutorial; it does not perform quality-aware consensus or
UMI correction.

# Real-data walkthrough

The [worked example](https://mutejester.github.io/AlignAIR/#/docs/worked-example) contains
the dataset details, download commands, checksums, recorded results, and interpretation. Its
recorded results came from AlignAIR 3 and its `alignair-igh-human@1.0.0` model; they await a
re-run with an AlignAIR 4 model. <!-- PENDING-V4-NUMBERS -->
This folder supplies [merge_pairs.py](merge_pairs.py), its demonstration read merger.

After downloading the two subsets into `aa_example/` as described in the guide, run
from the repository root:

```bash
python3 examples/end-to-end/merge_pairs.py aa_example/R1.sub.fastq aa_example/R2.sub.fastq aa_example/merged.fastq
alignair predict --input aa_example/merged.fastq --out aa_example/briney_merged.tsv \
  --model alignair4-igh-human
alignair validate-airr aa_example/briney_merged.tsv
alignair analyze aa_example/briney_merged.tsv --format text
```

`alignair4-igh-human` is the AlignAIR 4 human IGH model's id in the shared catalog. It is not published yet;
until it is, pass an AlignAIR 4 `.alignair` path to `--model`. <!-- PENDING-V4-PUBLISH: alignair4-igh-human -->

The merger drops pairs without an accepted overlap and writes placeholder quality scores.
It supports the reproducible tutorial; it does not perform quality-aware consensus or
UMI correction.

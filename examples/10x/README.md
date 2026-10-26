# 10x contigs with metadata

[filtered_contig.fasta](filtered_contig.fasta) contains four human IGH contigs.
[filtered_contig_annotations.csv](filtered_contig_annotations.csv) supplies their
Cell Ranger-style annotations, joined by `contig_id`.

Run from the repository root:

```bash
alignair predict --input examples/10x/filtered_contig.fasta --out tenx_airr.tsv \
  --model alignair-igh-human --metadata examples/10x/filtered_contig_annotations.csv \
  --metadata-id-column contig_id
```

The output retains the supported barcode, count, and chain metadata. These are IGH inputs;
light-chain or TCR contigs require a model for the corresponding locus.
See [integrations](https://mutejester.github.io/AlignAIR/#/docs/integrations) for metadata
selection and downstream workflows.

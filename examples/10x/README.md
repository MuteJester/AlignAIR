# 10x contigs with metadata

[filtered_contig.fasta](filtered_contig.fasta) contains four human IGH contigs.
[filtered_contig_annotations.csv](filtered_contig_annotations.csv) supplies their
Cell Ranger-style annotations, joined by `contig_id`.

Run from the repository root:

```bash
alignair predict --input examples/10x/filtered_contig.fasta --out tenx_airr.tsv \
  --model <v4-igh-model> --metadata examples/10x/filtered_contig_annotations.csv \
  --metadata-id-column contig_id
```

`<v4-igh-model>` is a placeholder for an AlignAIR 4 human IGH model (a `.alignair` path, a Hugging Face repo
`hf://org/repo`, or a catalog id); the id is not published yet. <!-- PENDING-V4-MODEL-ID -->

The output keeps the annotation's barcode, chain, count, cell and clonotype columns, and adds
their AIRR names (`cell_id`, `umi_count`, `consensus_count`, `c_call`). Metadata is carried as
given, not checked against the calls. These are IGH inputs; light-chain or TCR contigs require
a model for the corresponding locus.
See [integrations](https://mutejester.github.io/AlignAIR/#/docs/integrations) for metadata
selection and downstream workflows.

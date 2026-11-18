# 10x contigs with their cell metadata

[filtered_contig.fasta](filtered_contig.fasta) holds four human IGH contigs, and
[filtered_contig_annotations.csv](filtered_contig_annotations.csv) their Cell Ranger annotations (barcode, chain,
UMI and read counts, clonotype), one row per contig, keyed by `contig_id`.

Run from the repository root:

```bash
alignair predict --model alignair4-igh-human --input examples/10x/filtered_contig.fasta \
  --out tenx_airr.tsv --metadata examples/10x/filtered_contig_annotations.csv --metadata-id-column contig_id
```

The annotations are joined to the contigs by read id: the FASTA's ids are the table's `contig_id` values.
`--metadata-id-column contig_id` names that column; AlignAIR would also find it on its own, as the first column named
`sequence_id`, `contig_id`, `cell_id`, `id` or `name`.

`tenx_airr.tsv` has AlignAIR's 99 columns, then the annotation columns it keeps (`barcode`, `umis`, `reads`,
`raw_clonotype_id`, `raw_consensus_id`, `chain`, `is_cell`, `high_confidence`) and three of them again under their
AIRR names: `cell_id`, `umi_count` and `consensus_count`. Cell Ranger's `c_gene` fills AlignAIR's own `c_call`
column. The metadata is carried as given; it is not checked against the calls.

These contigs are IGH. For light-chain or TCR contigs, use the model of their locus: the annotation's `chain` column
tells them apart.

More on the join, `--keep-columns` and the `meta_` prefix:
[Join sample and cell metadata](https://mutejester.github.io/AlignAIR/#/docs/metadata-join).

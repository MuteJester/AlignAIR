# Real data, end to end

A public human IGH repertoire, from raw paired reads to a checked AIRR table, in four commands. This folder holds
[merge_pairs.py](merge_pairs.py), a small read-pair merger the example uses.

| | |
| --- | --- |
| Study | Briney et al. 2019, *Nature* 566:393-397, "Commonality despite exceptional diversity in the baseline human antibody repertoire" |
| Run | ENA/SRA `SRR8283604`, BioProject `PRJNA406949` |
| Reads | Human IGH, Illumina paired-end 2x250 |
| Subset | The first 2000 read pairs |

## 1. Download a subset

From the repository root, stream the first 2000 reads of each mate; `head` stops each download early, so you never
fetch the 1.2 GB files:

```bash
mkdir -p aa_example
B=https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR828/004/SRR8283604
curl -sSL "$B/SRR8283604_1.fastq.gz" | gzip -dc | head -n 8000 > aa_example/R1.sub.fastq
curl -sSL "$B/SRR8283604_2.fastq.gz" | gzip -dc | head -n 8000 > aa_example/R2.sub.fastq
md5sum aa_example/R1.sub.fastq aa_example/R2.sub.fastq
```

`curl` reports `(23) Failure writing output` when `head` stops it: that is expected. The checksums tell you the subset
is complete (they are this subset's, computed here, not ENA's checksums of the full files):

```text
f855e9a9488ac446f82d5e9fd8413bbe  aa_example/R1.sub.fastq
7cb75441bb2bf3821bd84c41656f00c7  aa_example/R2.sub.fastq
```

If a checksum differs, the connection dropped: run that line again.

## 2. Merge the read pairs

A full V(D)J amplicon spans both mates, and AlignAIR aligns single reads, so merge each pair first:

```bash
python3 examples/end-to-end/merge_pairs.py aa_example/R1.sub.fastq aa_example/R2.sub.fastq aa_example/merged.fastq
```

```text
pairs=2000 merged=1945
```

`merge_pairs.py` joins R1 to the reverse complement of R2 at their largest overlap of at least 20 nt with at most 10%
mismatches, drops the pairs that do not overlap (55 here), and writes placeholder quality scores. It needs nothing but
Python and gives the same file every time (md5 `06c7767cbe5718367d84d2fb624353c6`), which is all an example needs. It
does no quality-aware consensus and no UMI correction: for real work, use pRESTO's AssemblePairs, `fastp --merge` or
vsearch.

## 3. Align, check, summarise

```bash
alignair predict --model alignair4-igh-human --input aa_example/merged.fastq --out aa_example/briney_merged.tsv
alignair validate-airr aa_example/briney_merged.tsv
alignair analyze aa_example/briney_merged.tsv
```

```text
aligned 1945 reads (0 dropped) -> aa_example/briney_merged.tsv; 0 failed / 23 partial AIRR assemblies tagged
OK: aa_example/briney_merged.tsv — 1945 rows, no coordinate/CIGAR violations
```

`alignair analyze` prints a summary of the table (abridged):

```text
REPERTOIRE
  reads:            1945
  productive:       1769 (91.0%)
  V usage:         IGHVF10-G41*02 (257), IGHVF3-G5*01 (216), IGHVF6-G24*04 (114), ...
  J usage:         IGHJ4*02 (1204), IGHJ6*02 (253), IGHJ5*02 (237), ...
QC
  orientation:      reverse_complement 1931, complement 7, reverse 1, forward 6
```

What to notice:

- **Orientation.** Almost every read in this library is antisense: 1931 of 1945 come out with `rev_comp` = `T`.
  AlignAIR found and turned them itself, with no primer trimming and no orientation option.
- **Partial rows.** 23 rows are `partial` (`airr_assembly_status`): 17 reads without a V (`no_v`) and 6 whose indel
  breaks the reading frame (`nonproductive_indel`). They keep their other calls; filter to `complete` before junction
  or clone analysis.
- **Allele names.** The IGH model names alleles as OGRDB does (`IGHVF10-G41*02`), not with IMGT's names.
- **Raw reads.** These reads had no UMI consensus or error correction, so some sequencing errors show up as
  frameshifts or stop codons and make a read non-productive.

`validate-airr` checks the table's structure (columns, coordinate bounds, CIGARs). For the AIRR schema itself, the
`airr` Python package's `airr.validate_rearrangement` checks it too.

More: [Prepare your input](https://mutejester.github.io/AlignAIR/#/docs/prepare-input),
[Output columns explained](https://mutejester.github.io/AlignAIR/#/docs/output-columns) and
[How far to trust the calls](https://mutejester.github.io/AlignAIR/#/docs/trust-the-calls).

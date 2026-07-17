# End-to-end on real public data

A complete worked example on a real, public human IGH repertoire: download a subset, merge the read
pairs, predict, inspect the failures, validate the AIRR output, and summarise. Every number below is
captured from an actual run - nothing here is illustrative.

The `alignair` commands are checked against the real CLI in CI. The download and merge steps use only
`curl` and Python.

## The dataset

| | |
| --- | --- |
| Study | Briney et al. 2019, *Nature* 566:393-397, "Commonality despite exceptional diversity in the baseline human antibody repertoire" |
| Accession | ENA/SRA run **SRR8283604**, BioProject **PRJNA406949** |
| Material | Human immunoglobulin heavy chain (IGH) repertoire, Illumina paired-end (2x250) |
| License | INSDC record - sequence data is free to reuse without restriction |
| Model | `alignair-igh-human@1.0.0` (pinned) |

We use a small, reproducible subset: the first 2000 read pairs of the run.

## 1. Download a subset

Stream the first 2000 reads of each mate (no full 1.2 GB download - `head` stops the transfer early):

```bash
mkdir -p aa_example && cd aa_example
B="https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR828/004/SRR8283604"
curl -sSL "$B/SRR8283604_1.fastq.gz" | zcat | head -8000 > R1.sub.fastq
curl -sSL "$B/SRR8283604_2.fastq.gz" | zcat | head -8000 > R2.sub.fastq
md5sum R1.sub.fastq R2.sub.fastq
```

Expected checksums of the exact subset (so anyone reproduces the same bytes):

```
f855e9a9488ac446f82d5e9fd8413bbe  R1.sub.fastq
7cb75441bb2bf3821bd84c41656f00c7  R2.sub.fastq
```

If your network blocks the ENA HTTPS host, use the FTP URL instead (`ftp://ftp.sra.ebi.ac.uk/vol1/...`);
the subset and its checksums are identical either way.

## 2. Merge the read pairs

The reads are 2x250, so a full VDJ amplicon (~350-450 nt) spans both mates. Merge them with the
bundled zero-dependency merger ([`merge_pairs.py`](merge_pairs.py)):

```bash
python3 merge_pairs.py R1.sub.fastq R2.sub.fastq merged.fastq
md5sum merged.fastq
```

```
pairs=2000 merged=1945          # 97% of pairs overlap and merge
06c7767cbe5718367d84d2fb624353c6  merged.fastq
```

The merged reads are 398-444 nt - the whole rearrangement, end to end. (This merger is demo-grade; for
production use pRESTO, fastp, or vsearch, ideally with UMI consensus.)

## 3. Predict

```bash
alignair predict --input merged.fastq --out briney_merged.tsv --model alignair-igh-human@1.0.0
```

```
aligned 1945 reads (0 dropped) -> briney_merged.tsv; 0 failed / 43 partial AIRR assemblies tagged
```

Every read aligned, none failed. Two things worth noting already:

- **Orientation was handled for you.** 1937 of 1945 reads (99.6%) come out `rev_comp=T` - this library's
  reads are antisense, and AlignAIR's orientation head detected and re-framed every one, with no primer
  trimming and no orientation flag.
- **The calls use the model's reference nomenclature.** This model embeds the OGRDB human IGH set, so
  calls read like `IGHVF10-G41*03` (the OGRDB gene/allele naming), not the classic IMGT `IGHV3-23*01`.

## 4. Inspect the failures

43 of 1945 records assembled only partially:

```
assembly_status: complete 1902, partial 43
airr_assembly_reason: nonproductive_indel 42, collapsed_segment 1
```

These are not dropped - they keep their V/D/J calls, they just could not have a junction assembled
(an indel broke the reading frame). Filter to `airr_assembly_status == complete` before junction or
clonotype analysis.

## 5. Validate the AIRR output

```bash
alignair validate-airr briney_merged.tsv
```

```
OK: briney_merged.tsv — 1945 rows, no coordinate/CIGAR violations
```

A structural check: required columns present, coordinates in bounds, CIGARs consistent. For formal
AIRR-schema validation, also run `airr.validate_rearrangement("briney_merged.tsv")`.

## 6. Summarise

```bash
python3 - <<'PY'
import csv, collections
rows = list(csv.DictReader(open('briney_merged.tsv'), delimiter='\t'))
comp = [r for r in rows if r.get('airr_assembly_status') == 'complete']
print('complete:', len(comp), 'productive:', sum(r.get('productive')=='T' for r in comp))
print('top V:', collections.Counter(r['v_call'] for r in rows if r['v_call']).most_common(3))
for r in comp[:5]:
    if r.get('productive') == 'T':
        print(' ', r['v_call'], r['d_call'], r['j_call'], r['junction_aa'])
PY
```

Real CDR3s come straight out for the productive rearrangements:

```
IGHVF3-G6*01   IGHD6-13*01  IGHJ5*02  CAKNRVAAAGTMFATW
IGHVF10-G41*02 IGHD4-11*01  IGHJ5*02  CAKDSTLRLRYNWFDPW
```

Top V genes in the subset: `IGHVF10-G41*03` (256), `IGHVF3-G5*01` (215), `IGHVF6-G24*04` (113).

Of the 1902 complete records, 687 are productive. That fraction is lower than a polished repertoire
study reports, and honestly so: these are **raw reads** with no UMI-based error correction, merged with
a demo-grade tool, so uncorrected sequencing errors show up as frameshifts. AlignAIR reports them as
non-productive rather than hiding them - which is exactly the point of the assembly and productivity
flags. For functional repertoire analysis, apply UMI consensus and quality control upstream, then
filter to `productive == T`.

## What this shows

- Real public reads, no manual pre-orientation, no reference to configure: one command aligns them.
- AlignAIR's honesty signals (`rev_comp`, `airr_assembly_status`, `productive`) tell you what to trust.
- The output is standard AIRR TSV, ready for Change-O / Immcantation / Scirpy.

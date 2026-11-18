<p align="center">
  <img src="https://alignair.ai/_next/static/media/logo_alignair11bw.17e5d8d6.svg" width="200" alt="AlignAIR logo" />
</p>

<h1 align="center">AlignAIR</h1>

<p align="center">
  <b>V(D)J allele assignment for IG and TCR repertoires.</b><br>
  Reads in (FASTA, FASTQ or a table), an AIRR rearrangement table out, ready for Change-O, TIgGER and SCOPer.<br>
  Pretrained human models for IGH, IGK, IGL, TRA, TRB, TRD and TRG.
</p>

<p align="center">
  <a href="https://pypi.org/project/AlignAIR/"><img alt="PyPI" src="https://img.shields.io/pypi/v/AlignAIR"></a>
  <a href="https://pypi.org/project/AlignAIR/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
  <a href="https://mutejester.github.io/AlignAIR/"><img alt="Documentation" src="https://img.shields.io/badge/docs-website-6d5cf5"></a>
  <a href="https://doi.org/10.1093/nar/gkaf651"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1093%2Fnar%2Fgkaf651-blue"></a>
  <a href="LICENSE"><img alt="GPLv3" src="https://img.shields.io/badge/license-GPLv3-blue.svg"></a>
</p>

<p align="center">
  <a href="https://mutejester.github.io/AlignAIR/">Docs</a> ·
  <a href="https://mutejester.github.io/AlignAIR/#/docs/choose-a-model">Choose a model</a> ·
  <a href="https://mutejester.github.io/AlignAIR/#/docs/cli">Every option</a> ·
  <a href="https://doi.org/10.1093/nar/gkaf651">Paper</a>
</p>

---

## Install

```bash
pip install alignair
```

Python 3.10 or newer. A GPU is optional: AlignAIR uses CUDA or Apple MPS when it finds one, else the CPU. There is
no Bioconda package. A Docker image is published as `alignair/alignair` (the example reads are in `/app/examples`).

## Check it works

```console
$ alignair doctor
AlignAIR environment
  python           3.12.3
  torch            2.11.0
  cuda_available   False
  device_auto      cpu
  cache_dir        /home/you/.cache/alignair
```

(Abridged.) `device_auto` is where predictions run; `cache_dir` is where downloaded models are kept.

## Your first alignment

Download 20 example human IGH reads and align them with the IGH model:

<!-- PENDING-V4-PUBLISH: the release tag in this URL (v4.0.0) is set when the release is cut. -->
```bash
curl -LO https://raw.githubusercontent.com/MuteJester/AlignAIR/v4.0.0/examples/reads.fasta
alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv
```

The first run downloads the model from Hugging Face and caches it; later runs start at once. You get:

```text
aligned 20 reads (0 dropped) -> out.tsv
```

`out.tsv` is an AIRR rearrangement table, one row per read. Its first columns of interest:

```text
sequence_id  v_call                         d_call       j_call    junction_aa                  productive
read01       IGHVF10-G50*02                 IGHD3-10*01  IGHJ6*02  CATFYYGSGSYYNFVSNYNYYGMDVW   T
read02       IGHVF10-G42*01,IGHVF10-G42*03  IGHD2-2*02   IGHJ4*02  CAKGTSCNSTSCYTIDYW           T
read03       IGHVF3-G12*02                  IGHD5-18*02  IGHJ6*03  CARDTLNIVATPISAFAIYYYYYMDVW  T
```

When a read cannot tell V alleles apart, `v_call` lists them all, comma-separated (read02), as IgBLAST does. The IG
models name alleles as OGRDB does (`IGHVF10-G50*02`); see [Pretrained models](#pretrained-models). To align your own
reads, point `--input` at your file.

## In Python

```python
from alignair import Aligner, read_sequences

aligner = Aligner.from_pretrained("alignair4-igh-human")
ids, reads, info = read_sequences("reads.fasta")
result = aligner.predict(reads, ids=ids)
result.write_airr("out.tsv")
print(result.records[0]["v_call"], result.records[0]["junction_aa"])
```

```text
IGHVF10-G50*02 CATFYYGSGSYYNFVSNYNYYGMDVW
```

`result.records` holds one dict per read, in Python's terms (positions from 0, `True`/`False`); `write_airr` writes
the AIRR table, which is what tools downstream read.

## Pretrained models

| Locus | Model id | Germline set | Allele names | Alleles V / D / J |
|---|---|---|---|---|
| IGH | `alignair4-igh-human` | OGRDB human IGH (`HUMAN_IGH_OGRDB`) | OGRDB | 198 / 33 / 7 |
| IGK | `alignair4-igk-human` | OGRDB human IGK (`HUMAN_IGK_OGRDB`) | OGRDB | 168 / – / 8 |
| IGL | `alignair4-igl-human` | OGRDB human IGL (`HUMAN_IGL_OGRDB`) | OGRDB | 181 / – / 10 |
| TRA | `alignair4-tra-human` | IMGT human TRA (`HUMAN_TCRA_IMGT`) | IMGT | 76 / – / 68 |
| TRB | `alignair4-trb-human` | IMGT human TRB (`HUMAN_TCRB_IMGT`) | IMGT | 98 / 3 / 16 |
| TRD | `alignair4-trd-human` | IMGT human TRD (`HUMAN_TCRD_IMGT`) | IMGT | 14 / 3 / 4 |
| TRG | `alignair4-trg-human` | IMGT human TRG (`HUMAN_TCRG_IMGT`) | IMGT | 20 / – / 6 |

- **First use downloads the model** from Hugging Face into `~/.cache/alignair` (set `ALIGNAIR_CACHE_DIR` to move it).
  To work offline, download it ahead with `alignair models get alignair4-igh-human` and run with `--offline`.
- **Allele names.** The IG models call alleles by their OGRDB names (`IGHVF6-G21*07`), the TCR models by their IMGT
  names. A genotype written in the other naming is refused; the [Choose a model](https://mutejester.github.io/AlignAIR/#/docs/choose-a-model)
  page shows how to work with IMGT-named data.
- **Another germline set, or novel alleles,** for a species and locus a model covers: give the germline FASTA with
  `--reference`. **Another species** (mouse, say): [fine-tune or train a model](https://mutejester.github.io/AlignAIR/#/docs/do-i-need-to-train); binding
  another species' germlines to a human model costs accuracy.
- The models are released under GPL-3.0-or-later, as the code.

## Your own data

**Input:** FASTA, FASTQ, or a CSV/TSV table with a sequence column (`--sequence-column`), gzipped or not, or `-` for
standard input. Reads keep their ids; a repeated id gets a `_dup1` suffix. Reads longer than 575 nt are aligned on
their first 575 nt.

**Output:** an AIRR rearrangement TSV: all 99 columns by default, or a preset with `--columns airr|core|minimal`.
Beside it, `out.tsv.run.json` records the model, the reference and the options of the run.

**Common options:**

```bash
# Calls only from a donor's alleles: a YAML/JSON genotype, or the GenotypeSet `alignair genotype` writes
alignair predict --model alignair4-igh-human --input reads.fastq.gz --out out.tsv --genotype donor.yaml

# Your own germline set, novel alleles included
alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv --reference germline.fasta

# Carry 10x cell barcodes and UMI counts into the table. One locus per run: Cell Ranger's contigs mix chains,
# so split them by chain first (the metadata page shows how) and give each locus its own model.
alignair predict --model alignair4-igh-human --input igh_contigs.fasta --out igh.tsv \
  --metadata filtered_contig_annotations.csv

# Many reads on the CPU: worker processes
alignair predict --model alignair4-trb-human --input tcr.fasta --out tcr.tsv --threads 8
```

Every option, with its default, is on the [CLI reference](https://mutejester.github.io/AlignAIR/#/docs/cli) page;
`alignair predict --help` lists them by purpose.

## How it compares

On simulated reads held out from development (all seven loci, 13 test suites), AlignAIR is better than IgBLAST on
151 of 491 measure comparisons: the V mutation count (38 of 40), the J allele (25 of 49), segment boundaries, the exact
junction and clone key, the D allele, and calling no V on a read too short to show one. On real IGH repertoires of four
donors whose alleles are known from their genomes, the two tools' V calls are equally well supported, and AlignAIR
calls the reads with a known answer right more often (99.7% vs 99.5% on 5' RACE libraries). The
[comparison page](https://mutejester.github.io/AlignAIR/#/docs/comparison) shows every measure and the method.

**When another tool is the better choice:**

- **IgBLAST**, for a species or locus no AlignAIR model covers when you cannot train one; reads longer than 575 nt
  you cannot trim; a D call wherever any D bases remain (AlignAIR leaves a D with fewer than 8 bases uncalled);
  IgBLAST's scores and e-values; or a measure your study rests on where the
  [comparison](https://mutejester.github.io/AlignAIR/#/docs/comparison?h=when-igblast) says IgBLAST is better.
- **MiXCR**, for a pipeline from raw sequencing data to assembled clonotypes (UMIs, contig assembly): AlignAIR
  assigns V(D)J alleles to reads, it does not assemble clones.
- **Immcantation** stays your downstream toolkit: AlignAIR replaces its alignment step (AssignGenes and MakeDb), and
  Change-O, TIgGER and SCOPer read AlignAIR's table directly.

Known limits are listed on [How far to trust the calls](https://mutejester.github.io/AlignAIR/#/docs/trust-the-calls).

## Documentation

The website takes you further, by what you want to do:

- **Run on your data:** [your first alignment](https://mutejester.github.io/AlignAIR/#/docs/first-alignment),
  [prepare your input](https://mutejester.github.io/AlignAIR/#/docs/prepare-input),
  [donor genotypes](https://mutejester.github.io/AlignAIR/#/docs/donor-genotype),
  [your own germline reference](https://mutejester.github.io/AlignAIR/#/docs/own-reference).
- **Coming from another tool:** [IgBLAST](https://mutejester.github.io/AlignAIR/#/docs/from-igblast),
  [MiXCR](https://mutejester.github.io/AlignAIR/#/docs/from-mixcr),
  [Immcantation](https://mutejester.github.io/AlignAIR/#/docs/from-immcantation).
- **Look something up:** [every CLI option](https://mutejester.github.io/AlignAIR/#/docs/cli),
  [the Python API](https://mutejester.github.io/AlignAIR/#/docs/python-api),
  [every output column](https://mutejester.github.io/AlignAIR/#/docs/column-dictionary),
  [error codes](https://mutejester.github.io/AlignAIR/#/docs/error-codes).
- **Judge the results:** [how far to trust the calls](https://mutejester.github.io/AlignAIR/#/docs/trust-the-calls),
  [the comparison with IgBLAST](https://mutejester.github.io/AlignAIR/#/docs/comparison).
- **Train or fine-tune** a model for your species or germline set:
  [do I need to train?](https://mutejester.github.io/AlignAIR/#/docs/do-i-need-to-train),
  [train](https://mutejester.github.io/AlignAIR/#/docs/train),
  [fine-tune](https://mutejester.github.io/AlignAIR/#/docs/fine-tune).

## Cite

If you use AlignAIR, please cite:

> Thomas Konstantinovsky, Ayelet Peres, Ran Eisenberg, Pazit Polak, Ofir Lindenbaum, Gur Yaari.
> Enhancing sequence alignment of adaptive immune receptors through multi-task deep learning.
> *Nucleic Acids Research*, Volume 53, Issue 13, 22 July 2025, gkaf651.
> <https://doi.org/10.1093/nar/gkaf651>

```bibtex
@article{konstantinovsky2025alignair,
  title   = {Enhancing sequence alignment of adaptive immune receptors through multi-task deep learning},
  author  = {Konstantinovsky, Thomas and Peres, Ayelet and Eisenberg, Ran and Polak, Pazit and
             Lindenbaum, Ofir and Yaari, Gur},
  journal = {Nucleic Acids Research},
  volume  = {53},
  number  = {13},
  pages   = {gkaf651},
  year    = {2025},
  doi     = {10.1093/nar/gkaf651}
}
```

`CITATION.cff` holds the same reference, so GitHub's "Cite this repository" gives it too.

## Support

Questions and bugs: [GitHub issues](https://github.com/MuteJester/AlignAIR/issues). Contributions are welcome; see
[CONTRIBUTING](CONTRIBUTING.md).

## License

GPL-3.0-or-later (see [`LICENSE`](LICENSE)).

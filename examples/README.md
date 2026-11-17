# AlignAIR examples

Small input files, each with the commands that run on it. Run every command from the repository root. The
[AlignAIR documentation](https://mutejester.github.io/AlignAIR/#/docs/first-alignment) explains each step in full.

| Example | Files | On the website |
| --- | --- | --- |
| [First alignment](reads.fasta) | 20 productive human IGH reads simulated with GenAIRR (`make_example_reads.py` writes them) | [Your first alignment](https://mutejester.github.io/AlignAIR/#/docs/first-alignment) |
| [Donor genotype](donor_genotype/) | Two allele subsets of the IGH model's germline set | [Align with a donor genotype](https://mutejester.github.io/AlignAIR/#/docs/donor-genotype) |
| [10x contigs](10x/) | Four IGH contigs and their Cell Ranger annotations | [Join sample and cell metadata](https://mutejester.github.io/AlignAIR/#/docs/metadata-join) |
| [Reads in a table](airr/) | Reads in a TSV, and a sample metadata table | [Prepare your input](https://mutejester.github.io/AlignAIR/#/docs/prepare-input) |
| [Many samples](batch/) | A two-sample manifest and a Bash loop | [Run at scale](https://mutejester.github.io/AlignAIR/#/docs/run-at-scale) |
| [Train on your germline set](custom_reference/) | V, D and J germline FASTAs | [Train a model](https://mutejester.github.io/AlignAIR/#/docs/train) |
| [Compare two tools](compare/) | Two small hand-written AIRR tables | [Coming from IgBLAST](https://mutejester.github.io/AlignAIR/#/docs/from-igblast) |
| [Real data, end to end](end-to-end/) | A read-pair merger for a public human IGH data set | [Prepare your input](https://mutejester.github.io/AlignAIR/#/docs/prepare-input) |
| [From IgBLAST](igblast-migration/) | An AlignAIR run on the example reads | [Coming from IgBLAST](https://mutejester.github.io/AlignAIR/#/docs/from-igblast) |

## First run

<!-- PENDING-V4-PUBLISH: the model id alignair4-igh-human is confirmed when the models are published. -->
```bash
alignair doctor
alignair predict --model alignair4-igh-human --input examples/reads.fasta --out out.tsv
```

`alignair4-igh-human` is the human IGH model. The first run downloads it from Hugging Face and caches it; later runs
start at once. You get `aligned 20 reads (0 dropped) -> out.tsv`, and `out.tsv` is an AIRR rearrangement table with one
row per read. The IG models name alleles as OGRDB does (`IGHVF10-G50*02`), so the genotypes in these examples use
OGRDB names too.

To try AlignAIR with no download at all, `alignair demo` trains a tiny model on the spot and aligns a few simulated
reads with it.

[Choose a model](https://mutejester.github.io/AlignAIR/#/docs/choose-a-model) lists the seven pretrained models, and
the [CLI reference](https://mutejester.github.io/AlignAIR/#/docs/cli) every option.

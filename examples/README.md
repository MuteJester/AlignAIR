# AlignAIR examples

Sample inputs and short commands for the workflows documented on the
[AlignAIR website](https://mutejester.github.io/AlignAIR/#/docs/getting-started).
Run commands from the repository root unless an example says otherwise.

| Example | Included files |
| --- | --- |
| [Basic prediction](reads.fasta) | Eight simulated human IGH reads |
| [Donor genotype](donor_genotype/) | Two allele subsets for genotype-constrained prediction |
| [10x metadata](10x/) | Four IGH contigs and matching Cell Ranger-style annotations |
| [AIRR input](airr/) | A sequence table and sample metadata |
| [Multiple samples](batch/) | A two-sample manifest and a Bash loop |
| [Custom reference](custom_reference/) | V, D, and J germline FASTAs |
| [Tool agreement](compare/) | Two small, hand-written AIRR tables |
| [Real-data walkthrough](end-to-end/) | A demonstration paired-read merger |
| [IgBLAST migration](igblast-migration/) | An invocation using the included reads |

## First run

```bash
alignair doctor
alignair predict --input examples/reads.fasta --out out.tsv --model alignair4-igh-human
```

`alignair4-igh-human` is the AlignAIR 4 human IGH model's id in the shared catalog. It is not published yet;
until it is, pass an AlignAIR 4 `.alignair` path to `--model`. <!-- PENDING-V4-PUBLISH: alignair4-igh-human -->
A model named by catalog id or Hugging Face repo is downloaded on first use and cached. The
donor genotypes and the reads' truth labels use OGRDB allele names (`IGHVF1-G3*02`), as a model
trained on `HUMAN_IGH_OGRDB` does.
For an offline trial with a tiny model trained locally, run `alignair demo`.

See [installation](https://mutejester.github.io/AlignAIR/#/docs/getting-started),
[models](https://mutejester.github.io/AlignAIR/#/docs/models), and the
[CLI reference](https://mutejester.github.io/AlignAIR/#/docs/cli) for full options,
Docker commands, and output semantics.

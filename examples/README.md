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
alignair predict --input examples/reads.fasta --out out.tsv --model alignair-igh-human
```

The model is downloaded on first use. For an offline trial with a tiny model trained
locally, run `alignair demo`.

See [installation](https://mutejester.github.io/AlignAIR/#/docs/getting-started),
[models](https://mutejester.github.io/AlignAIR/#/docs/models), and the
[CLI reference](https://mutejester.github.io/AlignAIR/#/docs/cli) for full options,
Docker commands, and output semantics.

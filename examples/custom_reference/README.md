# Train on your own germline set

These human IGH FASTAs make a small germline set: [v.fasta](v.fasta) has eight V alleles (OGRDB names),
[d.fasta](d.fasta) six D alleles and [j.fasta](j.fasta) seven J alleles. The test suite uses them too.

Check the training plan from the repository root:

```bash
alignair train --v-fasta examples/custom_reference/v.fasta \
  --d-fasta examples/custom_reference/d.fasta --j-fasta examples/custom_reference/j.fasta \
  --chain-type BCR_HEAVY --out my_model --preset desktop --plan
```

`--plan` checks the germline set and the settings and prints the plan, without training. After GenAIRR's notes on
the germline set it built, you get:

```text
training plan:
{
  "loci": [
    "IGH"
  ],
  "alleles": {
    "V": 8,
    "D": 6,
    "J": 7
  },
  "has_d": true,
  "max_seq_length": 576,
  "model_parameters": 314812,
  "steps": 50000,
  "batch_size": 64
}
```

Remove `--plan` to train. The run writes its training snapshots (`my_model/model.alignair`, which can resume
training) and then the model to share and predict with, `my_model/bundle/model.alignair`:

```bash
alignair predict --model my_model/bundle/model.alignair --input examples/reads.fasta --out out.tsv
```

To align with a germline set of your own without training, give it to a pretrained model of the same locus with
`--reference`: [Use your own germline reference](https://mutejester.github.io/AlignAIR/#/docs/own-reference).
[Do I need to train?](https://mutejester.github.io/AlignAIR/#/docs/do-i-need-to-train) helps you choose, and
[Train a model](https://mutejester.github.io/AlignAIR/#/docs/train) covers the presets, other loci and what the
FASTAs need.

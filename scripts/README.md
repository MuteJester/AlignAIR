# DNAlignAIR scripts

Run everything with `PYTHONPATH=src .venv/bin/python scripts/<name> ...`.
The `benchmark` module (`python -m alignair.benchmark.cli`) is the canonical evaluator;
these scripts are research drivers and the model→benchmark adapter.

## Training
- `train_dnalignair.py` — train a DNAlignAIR checkpoint on a GenAIRR stream.
- `headtohead.py` — train, then score the model vs IgBLAST on identical GenAIRR strata
  (the research driver). Flags include `--reader`, `--scheduled-sampling`,
  `--reader-novel-prob` (simulated-novel reader augmentation), `--backbone shared`.

## Canonical benchmark (model → grade)
- `run_benchmark.py` — run a checkpoint over a benchmark case JSONL and emit prediction
  JSONL for `alignair.benchmark.cli evaluate` (feeds presented reads, full schema,
  optional `--calibration`). Pipeline: `cli build` → `run_benchmark.py` → `cli evaluate`
  → `cli assay`.
- `calibrate_sets.py` — fit the per-gene equivalence-set calibration (temperature + ε,
  `--objective f1`) on a representative GenAIRR mix → `allele_set_calibration.json`.
- `baseline_igblast.py` — IgBLAST bar only (needs IgBLAST in `.private/tools/`). Imported
  by several drivers for `gen_records`/`score`.

## Property-1 (dynamic genotype) evaluation
- `heldout_alleles.py` — genotype-subset compliance + novel-allele recall (strict/gene/
  equiv-class/set), with SNP-perturbed `~novel` stand-ins.
- `novel_source_test.py` — **definitive** novel test: edits reads so they genuinely derive
  from a novel germline (it is then the true closest), then measures recall.
- `embargo_retrain.py` — **gold-standard** Property-1 test: retrain with ~18 V alleles
  embargoed from BOTH the GenAIRR sim and the reference, then call them as novel against the
  full reference; reports held-out vs control (trained) allele recall.
- `hierarchical_eval.py` — value of hierarchical degradation + abstention on fragments
  (hard-error vs honest-abstain vs correct coarser call).
- `run_benchmark.py --genotype subset` exercises `genotype_mask_compliance`
  (outside_genotype_call_rate + genotype_restricted_call_acc) as a standing benchmark metric.

## Real-data validation (no simulation)
- `realdata_validation.py` — OAS real IGH reads: full-read concordance vs IgBLAST + the
  crop-back fragment test (IgBLAST full-read call = silver truth, test fragment recovery).
  Data lives gitignored under `.private/realdata/`.

## Diagnostics (one-off investigation drivers; kept for provenance)
- `segment_ablation.py` — oracle vs predicted segment → is segmentation the bottleneck?
- `retrieval_probe.py` — cosine vs MaxSim vs k-mer retrieval recall (falsified the
  late-interaction hypothesis: nothing beats cosine zero-shot).
- `reader_recall_test.py` — reader call accuracy vs candidate-net width (showed recall is
  not the heavy-SHM V cap; reader discrimination = SW parity is).
- `novel_diagnostic.py` — decomposes novel-allele recall into retrieval vs discrimination
  (showed the 0.33 was a metric artifact: retrieval is perfect, misses are real siblings).
- `diag_d_ambiguity.py` — D-allele ambiguity inspection.

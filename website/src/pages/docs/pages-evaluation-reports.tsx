import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const evaluationReports: DocPage = {
  slug: "evaluation-reports",
  title: "Evaluation reports API",
  section: "Design",
  lead: "Compute class and boundary reports from aligned observations, with explicit baseline and resampling conventions.",
  body: () => (
    <>
      <p>
        <code>alignair.evaluate</code> provides pure report and interval functions. They consume
        observed labels or numerical values supplied by an evaluator, independent of the model’s
        input alphabet. The shared observation collector applies task supervision masks and keeps
        source indices aligned. The evaluator selects the cohort and reporting policy. The{" "}
        <DocLink to="training-tasks">task interface</DocLink> defines the shared supervision rules.
      </p>
      <H2 id="collection">Collect supervised observations</H2>
      <CodeBlock lang="python" code={`from alignair.data.adapters import NucleotideBatchAdapter
from alignair.evaluate import collect_alignment_observations

adapter = NucleotideBatchAdapter(reference, model.cfg, device=device,
                                 augment_orientation=False)
observed = collect_alignment_observations(
    model, frozen_records, adapter, model.cfg, batch_size=128,
)
print(observed.record_count, observed.support)
for row in observed.classification["v"]:
    print(row.record_index, row.truth, row.prediction, row.correct)`} />
      <p>
        <code>collect_alignment_observations</code> uses the model configuration’s gene vocabulary
        and classification semantics. The same collector accepts <code>ProteinAlleleBatchAdapter</code>
        or <code>CategoricalBatchAdapter.from_config(config, device=device)</code>. Input alphabet
        and target preparation remain with the <DocLink to="training-batches">batch adapter</DocLink>.
        Optional <code>genes=</code> selects a nonempty unique subset of configured genes, for either
        categorical or multilabel models. The default selects every configured gene.
      </p>
      <DocTable head={["Observation", "Fields and meaning"]} rows={[
        ["record_count", "Total consumed source records, including records with unobserved fields."],
        ["classification[gene]", "ClassificationObservation rows with record_index, truth as a tuple of compatible IDs, prediction as a top-1 ID, and a membership-based correct property."],
        ["boundaries[gene_start / gene_end]", "BoundaryObservation rows with record_index, truth, prediction, and absolute_error in the adapter’s coordinate units."],
        ["support", "Separate observed classification and boundary counts; each gene/field retains its own denominator."],
      ]} />
      <p>
        Gene-level <code>{"{gene}_valid"}</code> and <code>{"{gene}_supervise"}</code> masks govern
        boundaries and classification. Categorical <code>{"{gene}_class_supervise"}</code> masks
        and unknown class IDs of <code>-1</code> can exclude classification while preserving
        observed boundaries. Compatible sets use the same validation as training tasks. Multilabel
        truth must be binary membership; an empty allele set has no classification observation.
        Compatible and multilabel sets are retained without choosing a canonical label. An exact-class
        summary needs singleton truth; broader sets can use each row’s membership correctness.
      </p>
      <p>
        The collector consumes an ordered iterable in bounded batches, includes a final partial
        batch, and retains detached Python values with zero-based source indices. The adapter must
        preserve order and cardinality and place inputs and targets on the model’s device. Fields
        are checked against configured vocabulary sizes and batch shapes. Non-finite supervised
        values raise <code>NumericalError</code>; malformed fields raise <code>TrainingDataError</code>
        with batch offset, gene, and field context. Masked placeholders do not count as observations.
        Invalid batch sizes, gene selections, or assembly options raise <code>ConfigurationError</code>.
      </p>
      <p>
        Inference disables gradients and restores every module’s prior training/evaluation flag,
        including on failures. Outputs and gradient graphs are not retained. Cohort freezing, source
        seeds, dataset fingerprints, and selection of auxiliary-head reports remain caller policies.
        Keep an indexable frozen source when later joining observations to study strata through
        <code> record_index</code>.
      </p>
      <p>
        Optional <code>prediction_set_threshold=0.5</code> retains sparse
        <code> prediction_sets[gene]</code> rows with <code>record_index</code> and
        <code> members</code>. This option requires multilabel probability heads and uses a strict
        greater-than comparison, including at thresholds zero and one. Rows align with observed
        classifications. Optional <code>scalar_heads=("mutation_rate", "productive")</code>
        collects named <code>ScalarObservation</code> rows with truth, prediction, and absolute
        error. Scalar heads accept row vectors or one-column tensors and honor their own
        <code> {"{head}_supervise"}</code> and <code>{"{head}_valid"}</code> masks. Their counts
        appear under <code>support["scalars"]</code>. Defaults retain neither optional payload;
        full probability matrices are never retained. Storage for requested sets grows with their retained membership.
      </p>
      <H2 id="alleles">Evaluate allele recovery and equivalence</H2>
      <CodeBlock lang="python" code={`from alignair.evaluate import evaluate_allele_cohort

# Choose a nucleotide or protein allele adapter for the same shared model config.
report = evaluate_allele_cohort(
    model, frozen_records, adapter, model.cfg, reference,
    class_maps={"V": v_allele_to_class, "J": j_allele_to_class},
    threshold=0.5, batch_size=128,
)
print(report["alleles"]["v"])
print(report["support"])`} />
      <p>
        <code>evaluate_allele_cohort</code> requires multilabel heads and a reference whose ordered
        allele vocabulary matches the model. The batch adapter binds targets to that order.
        Equivalence maps assign allele names to nonnegative integer class IDs; missing entries
        or <code>None</code> mean that the class is unavailable. Omit a gene’s map to report only
        its allele metrics. An empty map explicitly requests class reporting with zero mapped support.
      </p>
      <DocTable head={["Allele field", "Definition"]} rows={[
        ["n / nt_top1", "Observed allele rows and the fraction whose top prediction belongs to the truth set."],
        ["true_in_predicted_set / mean_pred_set_size", "Fraction with truth intersecting the thresholded predicted set, and mean predicted-set size."],
        ["class_top1 / class_support", "Class accuracy over rows whose prediction and every truth allele have mapped classes, plus that comparison count."],
        ["nt_error_rate", "Allele error fraction over all observed allele rows."],
        ["frac_errors_within_class / class_error_support", "Within-class fraction among allele errors with fully mapped class comparisons, plus that error count."],
        ["unmapped_class_rows", "Observed allele rows excluded from class comparison because a required class label is unavailable."],
      ]} />
      <p>
        Missing labels never match each other as a synthetic class. Class metrics can therefore
        have smaller denominators than allele metrics; inspect their coverage when comparing models.
        Zero support retains zero-valued rates with explicit zero counts. The report also contains
        <code> segmentation_mae</code>, <code>mutation_mae</code>, <code>productive_acc</code>,
        and per-field <code>support</code>. Each boundary or scalar metric uses its observed rows.
        Metadata heads are requested when <code>config.use_meta_heads</code> is enabled; disabled
        heads report null. An empty cohort reports <code>n=0</code>. Productivity truth is binary,
        and its prediction must be a probability.
      </p>
      <p>
        Reusable counters and normalization live in <code>evaluate.alleles</code>. Select
        <code> ProteinAlleleBatchAdapter(reference, config, device)</code> for translated reads;
        the former <code>aa.eval_gym</code> adapter has been removed. Cohort evaluation restores
        caller module modes. Boundary and scalar totals accumulate detached per-row
        errors, so older float32 batch reductions can differ by floating-point roundoff.
      </p>
      <H2 id="categorical-cohorts">Evaluate categorical cohorts with explicit strata</H2>
      <CodeBlock lang="python" code={`from alignair.data import CategoricalBatchAdapter
from alignair.evaluate import evaluate_categorical_cohort, protein_v_strata

adapter = CategoricalBatchAdapter.from_config(model.cfg, device=device)
report = evaluate_categorical_cohort(
    model, frozen_samples, adapter, model.cfg,
    genes=("v", "j"), batch_size=128,
    strata=protein_v_strata(class_maps["V"]),
)
print(report["classification"]["v"])
print(report["stratified"]["v_acc_by_coverage"])`} />
      <p>
        <code>evaluate_categorical_cohort</code> accepts either input alphabet and requires categorical
        heads. It reports every configured gene, including D when enabled, unless <code>genes</code>
        selects a subset. Its result contains <code>n</code>, <code>classification</code>,
        <code> boundaries</code>, <code>stratified</code>, and per-field <code>support</code>.
        Classification uses only supervised singleton truth; multi-member compatible truth raises
        <code> ConfigurationError</code>. Use observation membership correctness for a compatible-label study.
      </p>
      <p>
        Without strata, records may be an iterator and <code>stratified</code> is empty. Stratified
        evaluation requires an indexable frozen sequence so each supervised observation can be
        joined to its original record. Masked class rows never enter strata; their observed boundaries
        remain eligible. The report uses the same baselines, paired intervals, and exact-class
        denominators as the numerical functions below.
      </p>
      <CodeBlock lang="python" code={`from alignair.evaluate import ClassificationStratum

source_group = ClassificationStratum(
    name="v_acc_by_source", gene="v",
    group=lambda record, true_class: record["source_group"],
)
report = evaluate_categorical_cohort(
    model, frozen_samples, adapter, model.cfg, strata=(source_group,),
)`} />
      <p>
        A <code>ClassificationStratum</code> declares a unique report name, a selected gene, and a
        <code> group(record, true_class)</code> callback returning a nonempty string label. The callback
        receives the original record and singleton true class ID and must not mutate records.
        Each bin reports accuracy, count, and a Wilson interval. Invalid policies fail before
        the cohort is consumed; invalid returned labels include stratum, gene, and source index context.
      </p>
      <p>
        <code>protein_v_strata(class_map)</code> preserves the established V-study report keys and
        bins: visible residue spans <code>&lt;40</code>, <code>40-79</code>, and <code>≥80</code>, and
        synonymous allele counts <code>1</code>, <code>2</code>, and <code>≥3</code>. It expects V boundaries
        in amino-acid residue coordinates and does not translate nucleotide positions. Define a
        different grouping policy when units, bins, or scientific questions differ. Convergence
        boundary-only summaries omit these diagnostic strata while retaining their V/J gene selection.
      </p>
      <H2 id="classification">Report exact-class predictions</H2>
      <CodeBlock lang="python" code={`from alignair.evaluate import classification_report

report = classification_report(
    true_ids=[0, 0, 0, 1],
    predicted_ids=[0, 0, 1, 1],
    seed=0,
)
# micro_acc = 3/4; macro_acc = mean(2/3, 1) = 5/6
print(report["micro_acc"], report["macro_acc"])
print(report["paired_lift"])`} />
      <DocTable head={["Field", "Definition"]} rows={[
        ["micro_acc / micro_acc_ci95", "Fraction correct over observed rows, with a descriptive Wilson interval."],
        ["macro_acc", "Mean per-class accuracy, giving equal weight to each true class present."],
        ["majority_baseline", "Accuracy of the best constant class on this cohort; ties follow first observation order."],
        ["paired_lift", "Mean and 95% bootstrap interval of model correctness minus majority-predictor correctness on the same rows."],
        ["macro_over_micro / n_distinct_true_classes", "Macro-to-micro ratio and the number of observed true classes."],
      ]} />
      <p>
        Supply one-dimensional vectors of nonnegative integer class IDs with equal lengths.
        Unknown IDs such as <code>-1</code>, malformed labels, and mismatched lengths raise
        <code> InputError</code>. Select observed, singly labelled rows before calling this API.
        Compatible sets and multilabel truth require their own explicit correctness policy.
        The majority baseline is fitted to this cohort as a descriptive best-constant comparison;
        it is not a separately trained predictor. Standalone <code>macro_accuracy(y_true, y_pred)</code>
        and <code>majority_class_baseline(class_ids)</code> use the same rules.
      </p>
      <H2 id="boundaries">Report paired boundary errors</H2>
      <CodeBlock lang="python" code={`from alignair.evaluate import boundary_error_report

report = boundary_error_report(
    targets=[1.0, 2.0, 3.0, 10.0],
    absolute_errors=[1.0, 1.0, 0.0, 3.0],
)
# Model MAE = 1.25; constant-median MAE = 2.5.
print(report["relative_reduction"])  # 0.5
print(report["paired_improvement"])`} />
      <p>
        Supply finite, aligned target and absolute-error vectors in one coordinate unit. Errors
        must be nonnegative. The report includes <code>mae</code>, <code>mae_ci95</code>,
        <code> constant_median_baseline_mae</code>, <code>paired_improvement</code>, and
        <code> relative_reduction</code>. Each paired improvement is the constant predictor’s
        absolute error minus the model’s error on the same observation. The constant is the upper
        median for even cohorts, preserving the study convention. The standalone
        <code> constant_median_mae(values)</code> computes that baseline. Coordinate conversion
        and boundary supervision remain the evaluator’s responsibility.
      </p>
      <H2 id="intervals">Choose the interval convention</H2>
      <DocTable head={["Function", "Contract"]} rows={[
        ["wilson_interval(k, n, z=1.96)", "Descriptive binomial bounds. Counts must be integers with 0 ≤ k ≤ n; z must be positive and finite."],
        ["bootstrap_mean_ci(values, n_boot=1000, seed=0)", "Lower/upper NumPy percentile bounds using local default_rng draws and linear interpolation. Optional alpha defaults to 0.05."],
        ["paired_diff_ci(diffs, n_boot=2000, seed=0)", "A fixed 95% interval over paired per-observation advantages: mean, ci95, and lower_gt_zero."],
        ["bootstrap_competence_ci(values, n_boot=1000, alpha=0.05, seed=0)", "The historical gym convention: Python Random draws, sorted order-statistic endpoints, and a (mean, lower, upper) result."],
      ]} />
      <p>
        The NumPy and gym estimators deliberately retain their different seeded conventions.
        Neither changes the process random generators. Resampling counts must be positive integers,
        seeds nonnegative integers, and alpha strictly between zero and one. Invalid options raise
        <code> ConfigurationError</code>; malformed or non-finite observation vectors raise
        <code> InputError</code>.
      </p>
      <p>
        <code>bootstrap_mean_ci</code>, <code>paired_diff_ci</code>, and both report functions accept
        <code> max_resample_elements</code>, defaulting to 1,000,000. The NumPy estimator chunks
        resample matrices while preserving draw order and interval results. Each chunk holds at most
        that many elements, with a minimum of one complete resample. Input vectors and the
        <code> n_boot</code> sampled means need their own storage. The gym estimator already stores
        only its input and sampled means.
      </p>
      <p>
        Paired inputs must refer to the same observations. These estimators resample individual
        observations; class-stratified and lineage-cluster studies retain their distinct procedures.
        Empty vectors preserve historical zero-valued results. Retain the evaluator’s observation
        counts when interpreting them. Relative reduction is zero when its constant baseline is zero.
      </p>
      <H2 id="convergence-cohort">Convergence study cohorts</H2>
      <CodeBlock lang="python" code={`from alignair.aa.convergence.cohort import (
    COHORT_SCHEMA, build_cohort, cohort_fingerprint,
)

# Generation requires the source checkout's alignair_benchmark package.
cohort, fingerprint = build_cohort(n_per_stratum=40, seed=20250722, cal_frac=0.3)
assert cohort["schema"] == COHORT_SCHEMA
assert fingerprint == cohort_fingerprint(cohort)`} />
      <p>
        <code>convergence.cohort</code> owns this study’s declared strata, filtering, deterministic
        shuffle, calibration split and protein augmentations. Checkpoint scoring lives separately in
        <code> convergence.evaluator</code>; importing it does not require the source-only benchmark
        generator. Generation reports a contextual <code>MissingDependencyError</code> when that
        package is absent.
      </p>
      <p>
        <code>n_per_stratum</code> must be a positive integer, <code>seed</code> a nonnegative integer,
        and <code>cal_frac</code> a finite number strictly between zero and one. At least two eligible
        clean records are required. The shuffled split uses the rounded calibration count, bounded
        to leave at least one record in each slice. Calibration rows are excluded from the evaluation
        augmentation source. This is a record split; it does not deduplicate identical sequences.
        The existing clean/moderate separation, augmentation order and selection thresholds are retained.
      </p>
      <p>
        The <code>alignair.convergence.cohort.v2</code> fingerprint hashes canonical finite JSON for
        the complete returned cohort: ordered rows and declared strata, sequences, calls, D inversion,
        source records used for boundary projection, split parameters and the protein reference hash.
        Mapping insertion order is irrelevant. JSON encoding is streamed into SHA-256 without
        retaining a second serialized cohort. The driver records the schema beside the digest in
        <code> cohort_fingerprint.json</code> and the metrics artifact.
      </p>
      <p>
        These digests intentionally differ from the former sequence/call-only fingerprints.
        Historical artifacts are not rewritten. <code>build_cohort</code> and <code>STRATA_ORDER</code>
        are now imported from <code>alignair.aa.convergence.cohort</code>, rather than the evaluator.
        The returned dictionary remains mutable: a checksum describes its content at hashing time.
        Recompute it after edits, and retain separate model, configuration and reference bindings when
        comparing results. Fingerprinting alone does not validate scientific eligibility or freeze data.
      </p>
      <H2 id="ownership">Study integration</H2>
      <p>
        Numerical intervals live in <code>evaluate.statistics</code>; class and boundary summaries
        live in <code>evaluate.reports</code>. Convergence studies use the shared
        categorical evaluator with explicit V/J selection. The protein preset preserves historical
        coverage and ambiguity fields; no alphabet-specific evaluation implementation is required.
        Unknown and masked class rows are excluded from class summaries and strata while observed
        boundaries remain eligible. Multi-member truth sets require an explicit membership report.
        Convergence metrics share the macro-accuracy owner. Gym competence
        aggregation imports its preserved estimator directly; the historical
        <code> train.gym.instrument.stats.bootstrap_ci</code> alias remains available.
        The existing <code>alignair.evaluate</code> benchmark exports load on demand.
      </p>
      <p>
        The exploratory <code>alignair.aa.phase1.run_learning_proof</code> API has been retired,
        including its fixed seed ranges, held-out manifest/fingerprint helpers and
        <code> alignair.aa.phase1.v2</code> report. There is no replacement entry point for that
        report. Use the <DocLink to="training-workflow">shared training workflow</DocLink> for
        training, checkpoints and study callbacks, <code>alignair.train.overfit_one_batch</code>
        for capacity diagnostics, and the evaluator documented here for frozen-cohort reports.
        Cohort identity and train/evaluation separation remain explicit study responsibilities.
      </p>
      <H2 id="benchmark-contracts">Benchmark generation and model evaluation</H2>
      <p>
        The source-only <code>alignair_benchmark</code> package owns simulator case generation,
        source strata, orientation transforms, and cross-tool prediction scoring. Its predictor
        runners consume nucleotide case sequences and prediction dictionaries in the declared
        coordinate frame. The installed <code>alignair benchmark</code> command provides the
        separate four-stratum product self-check described in the
        <DocLink to="benchmarks"> benchmark guide</DocLink>.
      </p>
      <p>
        For raw model experiments, select an allele or categorical cohort evaluator above and
        supply the matching batch adapter. Protein studies retain forward-oriented cases whose
        records pass <code>alignair.data.projection.translated_record_usable</code>. Translation
        availability alone does not establish orientation eligibility, productivity, or full gene
        observability; apply the study’s declared cohort policy before evaluation.
      </p>
      <p>
        The unused <code>aa.benchmark.run_aa_benchmark</code> experiment runner has been removed.
        Its historical per-stratum <code>n_aa</code>/<code>no_aa_rate</code> JSON report has no
        drop-in replacement. Shared cohort reports describe observed classification, boundary,
        and scalar support; cross-tool benchmark reports use their own schema. Keep these
        denominators and coordinate contracts explicit when comparing results.
      </p>
      <H2 id="migration">Import shared evaluation utilities directly</H2>
      <p>
        The <code>aa.evaluate</code> and <code>aa.eval_gym</code> modules have been removed. Choose
        the shared evaluator and batch adapter explicitly. The categorical example above preserves
        the former frozen protein report’s V/J selection and diagnostic bins.
      </p>
      <DocTable head={["Former import", "Replacement"]} rows={[
        ["aa.evaluate.evaluate_frozen", "alignair.evaluate.evaluate_categorical_cohort with CategoricalBatchAdapter, V/J genes and protein_v_strata"],
        ["aa.eval_gym.evaluate_aa", "alignair.evaluate.evaluate_allele_cohort with ProteinAlleleBatchAdapter"],
        ["aa.evaluate._macro_accuracy", "alignair.evaluate.macro_accuracy"],
        ["aa.evaluate.majority_class_baseline / constant_median_mae", "alignair.evaluate (same function names)"],
        ["aa.evaluate.bootstrap_mean_ci / paired_diff_ci / wilson_interval", "alignair.evaluate (same function names)"],
        ["aa.eval_gym.gene_allele_counts / finalize_gene", "alignair.evaluate.alleles (same function names)"],
        ["aa.eval_gym._merge", "alignair.evaluate.alleles.merge_allele_counts"],
      ]} />
    </>
  ),
};

export const evaluationReportPages = [evaluationReports];

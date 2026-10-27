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
        seeds, dataset fingerprints, and optional auxiliary-head reports remain caller policies.
        Keep an indexable frozen source when later joining observations to study strata through
        <code> record_index</code>.
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
      <H2 id="ownership">Study integration</H2>
      <p>
        Numerical intervals live in <code>evaluate.statistics</code>; class and boundary summaries
        live in <code>evaluate.reports</code>. The protein frozen-cohort evaluator delegates those
        calculations and prediction collection while retaining its V/J selection, coverage bins,
        and ambiguity strata. Its existing metric fields remain; the added <code>support</code>
        field records per-field denominators. Unknown and masked class rows are excluded from class
        summaries and strata while observed boundaries remain eligible. Multi-member truth sets
        require an explicit membership report instead of this study’s exact-class summary.
        Convergence metrics share the macro-accuracy owner. Gym competence
        aggregation imports its preserved estimator directly. Historical estimator imports from
        <code> aa.evaluate</code> and <code>train.gym.instrument.stats</code> remain aliases.
        The existing <code>alignair.evaluate</code> benchmark exports load on demand.
      </p>
    </>
  ),
};

export const evaluationReportPages = [evaluationReports];

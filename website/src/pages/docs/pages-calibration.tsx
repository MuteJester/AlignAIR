import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const calibration: DocPage = {
  slug: "prediction-calibration",
  title: "Calibration API",
  section: "Design",
  lead: "Calibrate sigmoid allele confidence and categorical prediction sets through shared APIs for nucleotide and protein models.",
  body: () => (
    <>
      <p>
        <code>alignair.calibration</code> provides sigmoid temperature scaling and APS/RAPS class
        prediction sets. Choose the method that matches the model’s output semantics. Obtain raw
        probabilities through the <DocLink to="model-inference">shared inference API</DocLink> and
        retain their class-column order from the <DocLink to="reference-labels">reference label schema</DocLink>.
        Genotype threshold tuning remains a separate workflow in <code>alignair.genotype.calibrate</code>.
      </p>
      <DocTable head={["Method", "Inputs and purpose"]} rows={[
        ["Sigmoid temperature", "Independent allele probabilities in [0, 1], without a row-sum requirement. Fit binary top-1 correctness confidence, then transform sigmoid probabilities."],
        ["APS/RAPS", "Normalized categorical class probabilities and integer truth IDs. Fit a threshold for deterministic class prediction sets."],
      ]} />
      <H2 id="temperature">Sigmoid temperature scaling</H2>
      <CodeBlock lang="python" code={`import numpy as np
from alignair.calibration import apply_temperature, ece, fit_temperature

# Illustrative held-out top-1 confidences and binary correctness labels.
confidence = np.array([0.8, 0.8, 0.6, 0.6])
correct = np.array([1, 1, 1, 0])
temperature = fit_temperature(confidence, correct)
scaled = apply_temperature(confidence, temperature)
print(temperature, ece(confidence, correct), ece(scaled, correct))`} />
      <p>
        <code>apply_temperature(probs, temperature)</code> computes
        <code> sigmoid(logit(p) / temperature)</code>, clipping probabilities to [1e-6, 1 - 1e-6]
        before taking log odds. A finite positive temperature below one sharpens scores above 0.5
        and lowers scores below 0.5; a temperature above one moves scores toward 0.5.
        <code> None</code> or <code>1</code> returns the original input after validation. Transformed
        floating arrays keep their dtype; transformed integer arrays produce float64 probabilities.
      </p>
      <p>
        <code>fit_temperature(top1_probs, correct, lo=0.2, hi=3.0, n=120)</code> minimizes binary
        correctness NLL over a log-spaced grid, choosing the first temperature on a tie. It uses
        the same log-odds clipping and clips NLL probabilities to [1e-7, 1 - 1e-7]. Inputs are aligned,
        nonempty one-dimensional confidence and binary correctness vectors. This objective calibrates
        raw top-1 confidence; it does not optimize a multiclass softmax likelihood.
        <code> ece(top1_probs, correct, n_bins=10)</code> uses equal-width bins: the first includes
        both endpoints, and subsequent bins exclude the left endpoint and include the right.
        Empty aligned vectors return zero ECE.
      </p>
      <p>
        Clipping and finite precision can merge scores into ties. Fixed thresholds and relative-score
        set rules can also select different alleles after scaling. Check the downstream prediction
        policy; scaling does not guarantee unchanged calls or accuracy.
      </p>
      <H2 id="allele-fit">Fit per-gene allele temperatures</H2>
      <CodeBlock lang="python" code={`from alignair.calibration import calibrate_allele_temperatures

# model, reference and validation_records are supplied by the caller.
temperatures, report = calibrate_allele_temperatures(
    model, reference, validation_records,
    genes=("v", "j"), batch_size=128, min_samples=50, device="cpu",
)
print(temperatures)  # {gene: temperature}
print(report)        # {gene: {T, ece_before, ece_after, acc, n}}`} />
      <p>
        This workflow supports sigmoid allele heads for either input alphabet. Records supply
        <code> sequence</code> for nucleotide models or <code>sequence_aa</code> for protein models,
        plus comma-separated <code>v_call</code>, <code>d_call</code> or <code>j_call</code> truth.
        The raw top-1 prediction is correct when its reference name belongs to that truth set.
        Missing, empty and whitespace-only calls provide no observation; unmatched names count as
        incorrect. No boundary or mutation annotations are required.
      </p>
      <p>
        By default all configured genes are selected. An explicit selection must be a nonempty,
        unique subset, with ordered reference names matching each head’s width. Nucleotide inputs
        use the standard uppercase/unknown-base normalization and prefix crop. Protein inputs must
        be uppercase, use the supported amino-acid alphabet including X and *, and fit the model
        window without truncation. The complete cohort is validated before forwarding. Inference
        preserves module training modes and retains only top-1 confidence/correctness per gene,
        rather than full-cohort allele matrices.
      </p>
      <p>
        Genes with fewer than <code>min_samples</code> observed calls are omitted. An empty cohort
        returns two empty dictionaries. Report metrics are rounded to four decimals; returned
        temperatures retain full precision. <code>acc</code> describes raw top-1 correctness, and
        <code>n</code> counts that gene’s observed calls. The function does not change the model,
        its configuration or records, apply the fitted temperatures, or save artifacts. The caller
        owns held-out cohort selection, model/reference identity, evaluation and persistence.
      </p>
      <H2 id="fit">Fit APS/RAPS on a separate calibration cohort</H2>
      <CodeBlock lang="python" code={`import numpy as np
from alignair.calibration import Conformal, FrozenConformal

# Small illustrative arrays; production calibration needs a representative held-out cohort.
probabilities = np.array([
    [0.90, 0.06, 0.04], [0.90, 0.06, 0.04],
    [0.10, 0.05, 0.85], [0.10, 0.05, 0.85],
])
truth = np.array([0, 0, 1, 1])
calibrator = Conformal(alpha=0.1, min_group=2, seed=17).fit(
    probabilities, truth, groups=["long", "long", "short", "short"],
)
prediction = np.array([0.6, 0.3, 0.1])
print(sorted(calibrator.predict_set(prediction, group="long")))

# Apply the pooled threshold without fitting again.
frozen = FrozenConformal(calibrator.q_pooled, lam=calibrator.lam, k_reg=calibrator.k_reg)
print(sorted(frozen.predict_set(prediction)))`} />
      <DocTable head={["Parameter", "Contract"]} rows={[
        ["alpha", "Finite number strictly between zero and one."],
        ["lam / k_reg", "Nonnegative finite size penalty and nonnegative integer unpenalized rank count. lam=0 selects APS; a positive value adds the RAPS rank penalty."],
        ["min_group", "Positive integer minimum calibration-group size; default 200. Smaller and unseen groups use the pooled threshold."],
        ["seed", "Seed for the calibrator’s local NumPy generator. Successful fits advance its state; prediction draws no randomness."],
      ]} />
      <p>
        Probability matrices must be nonempty and real, with values in [0, 1] and row sums near one
        within numerical precision. Prediction accepts one probability vector. Inputs are validated
        without renormalization or changing their dtype. Raw logits and independent sigmoid scores
        must be converted through an appropriate probability interface first. Truth contains one
        integer class ID per row; multi-label or compatible-set calibration needs its own scoring policy.
      </p>
      <p>
        Groups must be aligned, comparable, non-null finite scalar labels. A successful refit replaces
        the pooled threshold and the complete group dictionary, including removal of old or newly
        undersized groups. Validation occurs before score sampling; thresholds are published after
        all quantiles are computed. Calling prediction before fitting or with a different class width
        raises a contextual error. Unknown groups fall back to the current pooled threshold.
      </p>
      <H2 id="arithmetic">Score and prediction conventions</H2>
      <p>
        <code>aps_scores(probs, true, rng, lam=0, k_reg=1)</code> ranks each row, sums probability mass
        through the true class, and subtracts one uniform draw times that class’s probability. RAPS
        adds <code>lam * max(0, rank - k_reg)</code> using one-based rank. The supplied generator must
        be a NumPy <code>Generator</code>.
      </p>
      <p>
        <code>conformal_quantile(scores, alpha)</code> retains the existing level
        <code> min(1, ceil((n + 1) * (1 - alpha)) / n)</code> and NumPy’s <code>method="higher"</code>.
        A level above one is clipped and returns the largest observed score; the implementation
        does not substitute an infinite threshold for that small-cohort case.
      </p>
      <p>
        <code>aps_set(prob_row, qhat, lam=0, k_reg=1)</code> returns the top-ranked prefix through the
        first cumulative, optionally penalized score reaching the threshold. The boundary class is
        included, a zero threshold retains the first class, and a threshold above every score returns
        all classes. Ranking retains NumPy’s existing default sort behavior, including ties. Prediction
        uses this deterministic prefix rule; it does not draw the random values used during fitting.
      </p>
      <H2 id="frozen">Stored calibration and product binding</H2>
      <p>
        <code>FrozenConformal(qhat, lam=0, k_reg=1)</code> applies a finite nonnegative stored threshold.
        The historical <code>from_calibration_json(cal, gene, group="full", lam=0, k_reg=1)</code>
        helper reads <code>genes[gene].calibration_frozen.q_mondrian[group]</code>; missing keys raise
        <code> KeyError</code>. It uses the supplied method parameters and does not infer them from
        another JSON block.
      </p>
      <p>
        The product loader still verifies model, class-map and calibration bindings and reads its
        stored method parameters. Product group selection and cross-gene pooled fallback retain
        their existing policy. These NumPy classes alone do not establish artifact identity or a
        deployment coverage claim. Keep training, calibration and evaluation data separate, preserve
        the label order, and assess coverage on the intended distribution under declared group definitions.
      </p>
      <H2 id="errors">Errors and migration</H2>
      <DocTable head={["Error", "Meaning"]} rows={[
        ["ConfigurationError", "Invalid calibration parameters, model profile, gene/reference binding, seed, or a request to predict before fitting."],
        ["InputError", "Invalid probabilities, observations, records, sequences, groups, or fitted class width. Normalization is required only by APS/RAPS."],
        ["NumericalError", "Non-finite probability or score values."],
        ["InferenceError", "Allele output width differs from the fitting reference; shared inference also checks output contracts."],
      ]} />
      <p>
        Numerical helpers attach <code>stage="temperature_calibration"</code> or
        <code> stage="conformal_calibration"</code> and the relevant field. Allele-fitting record and
        output checks attach stage, gene and source-position context where applicable. The former
        <code> alignair.aa.conformal</code> module has been removed. Import <code>Conformal</code>,
        <code> FrozenConformal</code>, <code>aps_scores</code>, <code>aps_set</code> and
        <code> conformal_quantile</code> from <code>alignair.calibration</code> or
        <code> alignair.calibration.conformal</code>. Existing JSON field names and valid single-fit
        numerical conventions are retained. The refit fix deliberately discards stale group thresholds.
      </p>
      <p>
        The former <code>alignair.predict.calibrate</code> module has also been removed. Import
        <code> apply_temperature</code>, <code>fit_temperature</code>, <code>ece</code> and
        <code> calibrate_allele_temperatures</code> from <code>alignair.calibration</code>.
        The scaling keyword is now <code>temperature</code>. Zero, negative and non-finite
        temperatures raise errors; use <code>None</code> or <code>1</code> for identity scaling.
        Existing valid floating-point arithmetic, default fitting grid and report keys are retained.
      </p>
    </>
  ),
};

export const calibrationPages = [calibration];

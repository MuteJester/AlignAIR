import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const calibration: DocPage = {
  slug: "prediction-calibration",
  title: "Prediction-set calibration API",
  section: "Design",
  lead: "Fit and apply APS/RAPS class prediction sets through one calibration interface for nucleotide and protein models.",
  body: () => (
    <>
      <p>
        <code>alignair.calibration</code> consumes categorical probability arrays and integer class
        labels. It does not select a model, tokenizer, cohort, or reference. Obtain probabilities
        through the <DocLink to="model-inference">shared inference API</DocLink> and retain their
        class-column order from the <DocLink to="reference-labels">reference label schema</DocLink>.
        Sigmoid temperature scaling remains in <code>predict.calibrate</code>; genotype threshold
        tuning remains in <code>genotype.calibrate</code>.
      </p>
      <H2 id="fit">Fit on a separate calibration cohort</H2>
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
        ["ConfigurationError", "Invalid calibration parameters, seed, or a request to predict before fitting."],
        ["InputError", "Invalid probability shape/range/normalization, truth IDs, groups, or fitted class width."],
        ["NumericalError", "Non-finite probability or score values."],
      ]} />
      <p>
        Errors carry <code>stage="conformal_calibration"</code> and the relevant field. The former
        <code> alignair.aa.conformal</code> module has been removed. Import <code>Conformal</code>,
        <code> FrozenConformal</code>, <code>aps_scores</code>, <code>aps_set</code> and
        <code> conformal_quantile</code> from <code>alignair.calibration</code> or
        <code> alignair.calibration.conformal</code>. Existing JSON field names and valid single-fit
        numerical conventions are retained. The refit fix deliberately discards stale group thresholds.
      </p>
    </>
  ),
};

export const calibrationPages = [calibration];

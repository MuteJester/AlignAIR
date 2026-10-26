import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const testing: DocPage = {
  slug: "testing",
  title: "Testing and validation",
  section: "Maintainer",
  lead: "Suite ownership, test selection, fixtures, and distribution checks.",
  body: () => (
    <>
      <H2 id="setup">Test environment</H2>
      <p>
        Run tests from the repository root. Pytest loads the source checkout using
        <code> pyproject.toml</code>. Follow the <DocLink to="getting-started">installation guide</DocLink>
        {" "}to choose a PyTorch build for your machine, then install the test dependencies:
      </p>
      <CodeBlock code={`python -m pip install -e ".[cli,test-research]"
python -m pytest`} />
      <p>
        The <code>test</code> extra supplies pytest, official AIRR validation, conda recipe rendering,
        coverage reporting, test timeouts, and parallel test workers.
        The <code>test-research</code> extra also supplies pyhmmer for HMM fixtures and scikit-learn for
        applicability scorers. The <code>dev</code> extra includes this complete test environment.
        Collector tests inject provider responses and numberers; they do not require live dataset or
        ANARCII model downloads.
      </p>

      <H2 id="layout">Suite responsibilities</H2>
      <DocTable head={["Location under tests/", "Coverage"]} rows={[
        [<code>alignair/</code>, "Main-package APIs and subsystems: CLI, I/O, reference, model files, prediction, training, genotype, registry, and errors."],
        [<code>alignair/aa/</code>, "AA data, models, training, evaluation, alignment, inference, convergence, Phase-3 scoring, and the prediction product."],
        [<code>alignair/aa/guard/</code>, "Source collection, cohort integrity, profiles, model components, training studies, and selection rules, each in a separate directory."],
        [<code>alignair_benchmark/</code>, "Core/artifact contracts, generation, I/O, evaluation, and CLI behavior. Shared case factories live in _support.py."],
        [<code>acceptance/</code>, "Fixed-reference guarantees, official AIRR validation, scientific release gates, and model-card claims tied to named tests."],
        [<code>robustness/</code>, "Malformed-input handling and mathematical invariants across package boundaries."],
        [<code>packaging/</code>, "Conda recipes, operation without optional runtime packages, compiled-wheel validation, and release smoke checks."],
        [<code>quality/</code>, "Checks that the QA infrastructure rejects missing evidence, altered model fixtures, external network calls, and inadequate branch coverage."],
        [<code>support/</code>, "Pytest policies, environment isolation, and explicit artifact/coverage tools. These helpers are not part of the installed package."],
        [<code>fixtures/training/</code>, "The shared IGH CSV used to check dataset shapes and allele-vocabulary cardinalities."],
        [<code>fixtures/models/validation.json</code>, "Versioned public IGH and IGK/IGL model identities, sizes, and SHA-256 checksums; model weights are fetched separately."],
      ]} />
      <p>
        Keep small fixtures beside their owning tests: the AIRR genotype schema, conformal fixture, and
        benchmark report snapshots are examples. Use the shared fixture directory for inputs consumed by
        multiple areas. Add subdirectories when an area gains a distinct responsibility.
      </p>

      <H2 id="selection">Run relevant checks</H2>
      <CodeBlock code={`# Source suite, including training smoke tests
python -m pytest tests

# Component contracts without expensive or model-backed checks
python -m pytest tests -m "component and not slow and not model_validation"

# Real provider/application integration
python -m pytest tests -m integration

# Reproduce an independence check with a stable shuffled order
python -m pytest tests -m "component and not slow and not model_validation" \\
  --test-order=shuffle --order-seed=20261026

# Exercise the same contracts in concurrent workers
python -m pytest tests -m "component and not slow and not model_validation" -n 2

# One subsystem
python -m pytest tests/alignair/aa/guard/training
python -m pytest tests/alignair_benchmark/evaluation

# Acceptance and malformed-input contracts
python -m pytest tests/acceptance tests/robustness -m "not local_data"

# Explicitly allow archived local research data
python -m pytest tests -m local_data --run-local-data

# Inspect discovery without executing cases
python -m pytest --collect-only -q`} />
      <p>
        Every test declares exactly one layer: <code>component</code>, <code>integration</code>, or
        <code> acceptance</code>. Additional markers describe capabilities: <code>slow</code>,
        <code> performance</code>, <code>model_validation</code>, and <code>local_data</code>.
        Archived local data requires <code>--run-local-data</code>; the default run skips it even when files
        exist. Model checks require the explicit cache described below. Markers and configuration are checked strictly.
      </p>

      <H2 id="models">Required model and release-claim validation</H2>
      <p>
        Fetch model fixtures before starting pytest. The manifest pins public artifact versions, sizes,
        and checksums. Provisioning reuses the package’s verified downloader; tests recheck the bytes and
        consume the cache offline. They never fall back to an arbitrary local model or download a replacement
        during execution.
      </p>
      <CodeBlock code={`python -m tests.support.model_artifacts --cache /tmp/alignair-test-models
python -m pytest tests -m "model_validation or acceptance or claim_evidence" \\
  --model-cache=/tmp/alignair-test-models --enforce-claims \\
  --claim-report=test-results/claims.json --junitxml=test-results/acceptance.xml`} />
      <p>
        The model-acceptance workflow runs these checks for CI and before release builds. Missing, skipped,
        expected-failing, or failed claim evidence fails the gate, including setup and teardown failures.
        The scientific check requires a complete seeded batch, matching simulator/model reference identities,
        finite metrics, and the version-controlled scientific thresholds. These synthetic regression checks
        establish published-model compatibility; biological performance claims also require held-out benchmark evidence.
        Claim enforcement runs serially so its report contains every required outcome; worker mode is rejected.
      </p>

      <H2 id="coverage">Coverage evidence</H2>
      <CodeBlock code={`python -m pytest tests --cov=alignair --cov=alignair_benchmark \\
  --cov-report=json:test-results/coverage.json --junitxml=test-results/junit.xml
python -m tests.support.coverage_gate --coverage=test-results/coverage.json`} />
      <p>
        CI saves JUnit, XML/JSON coverage, and HTML coverage reports. On Python 3.12 it enforces separate
        statement and branch floors for the critical files listed in <code>tests/coverage-policy.json</code>.
        The initial floors reflect measured coverage of error handling, sequence reading, model containers,
        registry caching, simulator coordinates, training loss, and convergence evaluation. A missing file
        in the report fails the check. Review threshold changes together with the behavior they protect.
      </p>

      <H2 id="writing">Test contracts and fixtures</H2>
      <p>
        Use deterministic inputs and assert observable behavior: values, shapes, coordinates, saved artifact
        contents, or the exception and diagnostic raised for a bad input. Exercise interactions through
        public entry points where practical. Inject provider responses rather than depending on remote data.
        Follow the <DocLink to="errors">errors and diagnostics API</DocLink> for error contracts.
      </p>
      <p>
        Keep fixtures in the nearest shared <code>conftest.py</code>. Root fixtures provide
        <code> repo_root</code> and <code>training_csv</code>; write generated outputs to <code>tmp_path</code>.
        Import helpers from support modules rather than another test module.
        Update <code>alignair.validation.gates.CLAIM_TESTS</code> when moving referenced tests.
        Scientific release metrics must be present and finite before their thresholds are evaluated.
      </p>
      <p>
        Each test receives isolated cache/config directories and a stable random seed derived from its
        node ID. Python, NumPy, and PyTorch random state and relevant Torch settings are restored afterward.
        Pin explicit seeds for scientific experiments whose identity must survive a test rename.
        External Python network access is rejected; loopback remains available for local protocol tests.
        Tests and fixtures have a 300-second timeout, with additional CI job limits.
      </p>
      <p>
        Evaluator component fixtures are independent of live cohort generation and include hand-calculated
        expected metrics. Provider integration checks remain separate. Use fresh mutable inputs for each
        case and release processes, temporary files, and tracing resources even when an assertion fails.
      </p>
      <p>
        Refresh benchmark snapshots by running their owning test file with <code>--update-goldens</code>.
        Snapshot refresh is forbidden in CI. Review the resulting JSON changes before accepting them.
      </p>

      <H2 id="artifacts">Distribution checks</H2>
      <p>
        Cibuildwheel runs <code>tests/packaging/wheel_smoke.py</code> against each built wheel to load and
        execute the compiled CIGAR kernel. CI and release workflows run
        <code> tests/packaging/release_smoke.sh</code> against installed wheels and Docker images.
        It checks version reporting, environment diagnostics, a tiny training/prediction demo,
        AIRR validation, and comparison output.
        The wheel environment also runs <code>tests/packaging/installed_contract.py</code> outside the checkout,
        exercising safe model serialization, reload, prediction, AIRR writing, offline loading, typed errors,
        and the shipped self-evaluator. It checks that the research benchmark harness remains excluded and rejects
        imports from the source tree.
      </p>
      <CodeBlock code={`sh tests/packaging/release_smoke.sh /tmp/alignair-release-check`} />
      <p>
        Set <code>ALIGNAIR</code> to select a particular installed executable. Source tests and distribution
        checks exercise different installations; both are required before a release.
      </p>
    </>
  ),
};

export const testingPages: DocPage[] = [testing];

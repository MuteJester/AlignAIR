import { CodeBlock, Callout } from "../../components/ui";
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
        <code> pyproject.toml</code> (with <code>src</code> on the path, so the development packages
        <code> alignair_assay</code> and <code>alignair_lab</code> are importable). Follow the
        <DocLink to="first-alignment"> installation guide</DocLink> to choose a PyTorch build for your machine, then
        install the test dependencies:
      </p>
      <CodeBlock code={`python -m pip install -e ".[cli,test]"
python -m pytest`} />
      <p>
        The <code>test</code> extra supplies pytest, official AIRR validation, conda recipe rendering, coverage
        reporting, test timeouts, parallel test workers, and SciPy for the architecture lab&apos;s noise statistics.
        The <code>dev</code> extra adds build, upload and wheel-building tools to the same environment.
      </p>

      <H2 id="layout">Suite responsibilities</H2>
      <DocTable head={["Location under tests/", "Coverage"]} rows={[
        [<code>alignair/</code>, "Main-package APIs and subsystems, one directory each: CLI, I/O, reference, genotype, model files, network, refinement, prediction, tasks, training, registry, and errors."],
        [<code>alignair_assay/</code>, "The development evaluation harness: observable truth, AIRR scoring, metrics and statistics, suites, runners, reports, and its CLI."],
        [<code>alignair_lab/</code>, "The architecture lab: recipes, record providers, screens, noise statistics, and replicates."],
        [<code>acceptance/</code>, "Public guarantees: official AIRR validation of golden output, the reference contract (a novel allele given in the reference is called), and identical batches from data workers."],
        [<code>robustness/</code>, "Malformed-input handling and invariants across package boundaries, such as the orientation involution."],
        [<code>packaging/</code>, "Conda recipes, dependency pins, operation without optional runtime packages, compiled-wheel validation, and release smoke checks."],
        [<code>quality/</code>, "Checks that the QA infrastructure rejects missing claim evidence, altered model fixtures, leaked process state, and inadequate branch coverage, and that retired namespaces and the development packages stay out of the product."],
        [<code>support/</code>, "Pytest policies, environment isolation, the release-claim map, and artifact and coverage tools. These helpers are not part of the installed package."],
        [<code>fixtures/models/validation.json</code>, "Versioned public model identities, sizes, and SHA-256 checksums for the model-validation cache; model weights are fetched separately."],
      ]} />
      <p>
        Keep small fixtures beside their owning tests. Use a shared fixture directory for inputs consumed by multiple
        areas, and add subdirectories when an area gains a distinct responsibility.
      </p>

      <H2 id="selection">Run relevant checks</H2>
      <CodeBlock code={`# Source suite
python -m pytest tests

# Component contracts without expensive or model-backed checks
python -m pytest tests -m "component and not slow and not model_validation"

# Real simulator, process, and serialization integration
python -m pytest tests -m integration

# Reproduce an independence check with a stable shuffled order
python -m pytest tests -m "component and not slow and not model_validation" \\
  --test-order=shuffle --order-seed=20261026

# Exercise the same contracts in concurrent workers
python -m pytest tests -m "component and not slow and not model_validation" -n 2

# One subsystem
python -m pytest tests/alignair/train
python -m pytest tests/alignair_assay/scoring

# Acceptance and malformed-input contracts
python -m pytest tests/acceptance tests/robustness

# Inspect discovery without executing cases
python -m pytest --collect-only -q`} />
      <p>
        Every test declares exactly one layer: <code>component</code>, <code>integration</code>, or
        <code> acceptance</code>; collection fails otherwise. Additional markers describe capabilities:
        <code> slow</code>, <code>performance</code>, <code>model_validation</code>, and <code>claim_evidence</code>
        (added automatically to every test a release claim names). Markers and configuration are checked strictly.
      </p>

      <H2 id="models">Release-claim validation</H2>
      <p>
        <code>tests.support.claims.CLAIM_TESTS</code> maps each release claim of the model card to the named test that
        proves it, and <code>tests/quality/test_claims.py</code> checks that every entry names a real test. With
        <code> --enforce-claims</code>, a run fails when any claim&apos;s test was not selected or did not pass setup,
        call and teardown; <code>--claim-report</code> writes the outcomes as JSON. Claim enforcement runs serially, so
        worker mode is rejected.
      </p>
      <CodeBlock code={`python -m tests.support.model_artifacts --cache /tmp/alignair-test-models
python -m pytest tests -m "model_validation or acceptance or claim_evidence" \\
  --model-cache=/tmp/alignair-test-models --enforce-claims \\
  --claim-report=test-results/claims.json --junitxml=test-results/acceptance.xml`} />
      <p>
        The model-acceptance workflow runs these commands. <code>tests.support.model_artifacts</code> fetches the models
        the manifest pins through the package&apos;s verified downloader, and tests consume the cache offline after
        rechecking each file&apos;s size and checksum; they never fall back to another local model or download a
        replacement. These are synthetic regression checks; biological performance claims also require held-out
        benchmark evidence.
      </p>
      {/* PENDING-V4-PUBLISH: re-pin tests/fixtures/models/validation.json to the alignair4-* models. */}
      <Callout kind="note" title="Fixtures are re-pinned at release">
        The manifest (<code>fixtures/models/validation.json</code>) still pins the AlignAIR 3 catalog models. It is
        re-pinned to the AlignAIR 4 models (<code>alignair4-igh-human</code> and the other loci) once they are
        published.
      </Callout>

      <H2 id="coverage">Coverage evidence</H2>
      <CodeBlock code={`python -m pytest tests --cov=alignair \\
  --cov-report=json:test-results/coverage.json --junitxml=test-results/junit.xml
python -m tests.support.coverage_gate --coverage=test-results/coverage.json`} />
      <p>
        CI saves JUnit, XML/JSON coverage, and HTML coverage reports. On Python 3.12 it enforces separate statement
        and branch floors for the critical files listed in <code>tests/coverage-policy.json</code>: the model and its
        components, model files, sequence I/O and input adapters, the registry cache, prediction stages, errors, batch
        adapters and targets, training setup, engine, loop, checkpoints and record streams, tasks and metrics, and the
        novel-allele path. A listed file missing
        from the report fails the check. Review threshold changes together with the behavior they protect.
      </p>

      <H2 id="writing">Test contracts and fixtures</H2>
      <p>
        Use deterministic inputs and assert observable behavior: values, shapes, coordinates, saved artifact contents,
        or the exception and diagnostic raised for a bad input. Exercise interactions through public entry points where
        practical. Inject provider responses rather than depending on remote data. Follow the
        <DocLink to="errors"> errors and diagnostics API</DocLink> for error contracts.
      </p>
      <p>
        Keep fixtures in the nearest shared <code>conftest.py</code>. The root fixture <code>repo_root</code> gives the
        repository path; write generated outputs to <code>tmp_path</code>. Import helpers from support modules rather
        than another test module. Update <code>tests.support.claims.CLAIM_TESTS</code> when moving a test a claim
        names.
      </p>
      <p>
        Each test receives isolated cache and config directories (AlignAIR, XDG and Hugging Face) and a stable random
        seed derived from its node ID. Python, NumPy, and PyTorch random state, the default dtype, gradient mode,
        deterministic-algorithm settings and the Torch thread count are restored afterward; the suite runs Torch on 2
        threads. Pin explicit seeds for experiments whose identity must survive a test rename. External network access
        and DNS lookups fail the test; loopback remains available for local protocol tests. Tests and fixtures have a
        300-second timeout, with additional CI job limits. Use fresh mutable inputs for each case, and release
        processes and temporary files even when an assertion fails.
      </p>

      <H2 id="artifacts">Distribution checks</H2>
      <p>
        Cibuildwheel runs <code>tests/packaging/wheel_smoke.py</code> against each built wheel to load and execute the
        compiled germline-CIGAR and V re-score kernels. CI and release workflows run
        <code> tests/packaging/release_smoke.sh</code> against installed wheels and Docker images. It checks version
        reporting, environment diagnostics, a tiny training/prediction demo with its model bundle, AIRR validation, and
        a comparison of default and donor-reference output. The wheel environment also runs
        <code> tests/packaging/installed_contract.py</code> outside the checkout, exercising pickle-free model
        serialization, reload, prediction, AIRR writing, offline loading, typed errors, and the shipped self-evaluator.
        It checks that the architecture lab is not installed and rejects imports from the source tree.
      </p>
      <CodeBlock code={`sh tests/packaging/release_smoke.sh /tmp/alignair-release-check`} />
      <p>
        Set <code>ALIGNAIR</code> to select a particular installed executable. Source tests and distribution checks
        exercise different installations; both are required before a release.
      </p>
    </>
  ),
};

export const testingPages: DocPage[] = [testing];

import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const aaGuardResearch: DocPage = {
  slug: "aa-guard-research",
  title: "AA guard research API",
  section: "Maintainer",
  lead: "Source collection, cohort auditing, profile construction, and training studies for the amino-acid applicability guard.",
  body: () => (
    <>
      <p>
        These research interfaces prepare and evaluate applicability-guard candidates. Run them from a source
        checkout with <code>PYTHONPATH=src</code>, supplying the input and output paths for each study.
        They do not publish models or establish a release decision. For everyday prediction, use the{" "}
        <DocLink to="python-api">Python API</DocLink> or <DocLink to="cli">CLI</DocLink>.
      </p>

      <H2 id="modules">Module responsibilities</H2>
      <DocTable head={["Module under alignair.applicability", "Responsibility"]} rows={[
        [<code>data_sources</code>, "Source-specific collection, exclusions, verified download caches, and optional independent domain numbering."],
        [<code>cohort.run_gate</code>, "Allocation, structural-negative generation, post-allocation deduplication, and the cohort audit."],
        [<code>disjoint</code>, "Reproduce the historical pre-split corpus digest and audit fresh-record overlaps without opening the sealed partition."],
        [<code>profiles.alignment</code>, "Germline allele selection, translation, protein deduplication, and alignment."],
        [<code>profiles.builder.build_profile_bank</code>, "Build a complete HMM bank from checksum-pinned local sources."],
        [<code>feature_cache</code>, "Prepare and validate HMM features, retain cache identities in memory, and publish persistent caches atomically."],
        [<code>training.manifests</code>, "Bind explicit study settings to data, profile, partition, and fold identities."],
        [<code>training.studies</code>, "Run seed sweeps and leave-one-category-out training."],
        [<code>training.sampling</code>, "Validate training pools, allocate complete batches, and checkpoint the sampling policy, RNG and rotation position."],
        [<code>checkpoint</code>, "Verify guard runtime identities and training contracts; load and atomically publish trusted local checkpoints."],
        [<code>training.budget</code>, "Compute accounting using a caller-supplied ledger path."],
        [<code>training.choose_production_seed</code>, "Choose the lowest development loss, breaking ties by the lowest seed; reject fresh-validation metrics."],
      ]} />

      <H2 id="sources">Collect source records</H2>
      <p>
        Collection commands verify exclusions using <code>--cleaned</code>, <code>--fingerprint</code>, and{" "}
        <code>--phase3-gold</code>. Supply <code>--store</code> for raw caches and normalized records and{" "}
        <code>--census-dir</code> for source reports. Raw sequences and research artifacts belong in local study
        directories. ANARCII is loaded when domain numbering runs; profile building and loading require pyhmmer.
      </p>
      <CodeBlock code={`python -m alignair.applicability.data_sources.oas --help
python -m alignair.applicability.data_sources.genairr --help
python -m alignair.applicability.data_sources.igsf --help
python -m alignair.applicability.data_sources.sabdab --help
python -m alignair.applicability.data_sources.tenx_tcr --help`} />
      <DocTable head={["Collector", "Inputs and behavior"]} rows={[
        ["OAS", <><code>probe</code> builds a catalog; <code>unpaired</code> uses <code>--catalog</code>.
          The <code>paired</code> and <code>diverse</code> commands retain separate baseline and donor-diversity sampling policies.
          Prefix caches bind request identity, byte limits, and checksums; incomplete CSV rows are excluded.</>],
        ["GenAIRR", "Preserves the simulated-positive study settings, record identities, and genairr_sim output filenames."],
        ["IgSF", <>Also requires <code>--previous-records</code>. Uses UniProt/InterPro domain coordinates;
          reused raw caches require matching checksum sidecars.</>],
        ["SAbDab", <>Also requires <code>--previous-records</code>.
          Retains the VHH/murine sampling limits and sequence deduplication; domain extraction loads ANARCII lazily.</>],
        ["10x TCR", <>Uses the four declared dataset specifications unless <code>--units</code> supplies a JSON list.
          Raw downloads are verified; numbering and the current denylist are reapplied on every run.
          Old normalized-unit caches do not bypass exclusions.</>],
      ]} />
      <p>
        Use each command’s <code>--help</code> for its complete argument list. Source outputs retain their
        established normalized-record and census filenames. The historical Gate-3 corpus is an input to digest
        verification; collecting a new cohort does not regenerate that corpus.
      </p>

      <H2 id="cohort">Audit and freeze a cohort</H2>
      <CodeBlock code={`python -m alignair.applicability.cohort \\
  --store study/sources --census-dir study/census \\
  --fingerprint inputs/split_fingerprint.json --out study/cohort

python -m alignair.applicability.disjoint fresh.jsonl \\
  --cleaned inputs/cleaned_cohort.jsonl \\
  --fingerprint inputs/split_fingerprint.json --phase3-gold inputs/gold_labels.tsv`} />
      <p>
        The cohort gate writes <code>STAGE1_RESULT.json</code> and freezes{" "}
        <code>FROZEN_COHORT_MANIFEST.json</code> only on GO. A stopped audit returns a nonzero exit status.
        Inputs must remain unchanged during the audit, and an existing frozen manifest cannot be replaced.
        The gate binds normalized-source snapshots and the supplied fingerprint; source collectors perform
        the upstream denylist filtering.
      </p>
      <p>
        The disjointness command checks sequence, lineage, donor, study, and Phase-3 therapeutic overlap.
        It reports counts and returns nonzero for overlaps or invalid inputs.
      </p>

      <H2 id="profiles">Build a pinned profile bank</H2>
      <CodeBlock code={`python -m alignair.applicability.profiles --sources sources.json --out study/profile-bank`} />
      <p>
        The source JSON must contain <code>imgt_gene_db</code>, <code>ogrdb_human_igh</code>,{" "}
        <code>ogrdb_human_igk</code>, <code>ogrdb_human_igl</code>, and <code>pfam_pf07686</code>.
        Each entry supplies a local <code>path</code>, provenance <code>url</code>, release <code>version</code>,
        and expected <code>sha256</code>. Relative paths resolve against the source JSON’s directory.
      </p>
      <CodeBlock lang="json" code={`{
  "path": "raw/ogrdb_igh.fasta",
  "url": "https://example.org/pinned-release",
  "version": "release-id",
  "sha256": "<64 lowercase hexadecimal characters>"
}`} />
      <p>
        This example is one source entry. Supply the IMGT WithGaps FASTA, OGRDB published/gapped FASTAs, and
        compressed Pfam PF07686 HMM as local files. The builder verifies every checksum, builds all required
        profiles, and writes HMMs, amino-acid alignments, and <code>profile_bank_manifest.json</code> into a new
        directory. Raw source files stay in place. Missing profiles and existing output directories are refused;
        the builder does not download replacement inputs. Existing imports from <code>alignair.applicability.profiles</code> remain available.
      </p>

      <H2 id="training">Prepare and run training studies</H2>
      <CodeBlock code={`python -m alignair.applicability.training freeze-sweep --help
python -m alignair.applicability.training freeze-loco --help
python -m alignair.applicability.training sweep --help
python -m alignair.applicability.training loco --help`} />
      <DocTable head={["Command", "Required inputs"]} rows={[
        [<code>freeze-sweep</code>, <><code>--settings --records --profile-manifest --fresh-sources --checkpoints --out</code></>],
        [<code>freeze-loco</code>, <><code>--training-manifest --runs --records --profile-manifest --checkpoints --out</code></>],
        [<code>sweep</code>, <><code>--manifest --records --profiles --out --feature-cache</code></>],
        [<code>loco</code>, <><code>--manifest --training-manifest --records --profiles --out --feature-cache</code></>],
      ]} />
      <p>
        Sweep settings include the candidate/seed program, optimizer, schedule, training configuration, loss
        weights, energy margins, temperatures, GRL coefficient, OE/mining policies, and production-seed rule.
        A previous sweep manifest can supply settings; preparation recomputes data identities and counts.
        Missing or malformed fresh-exclusion sources stop preparation. For preparation commands,
        <code> --out</code> names a new manifest file; for training commands it names the report and accounting directory.
      </p>
      <p>
        LOCO preparation verifies the frozen sweep and restricts its existing partition for each held-out group
        without reshuffling. It fixes seeds from development losses and binds profile masks, source vocabularies,
        and excluded-record hashes per fold. Preparation does not load HMMs or checkpoints, start training,
        or open challenge data. Existing manifest files cannot be replaced.
      </p>
      <p>
        Manifests and study execution share
        <code> alignair.data.partitioning.partition_by_parent_lineage</code>, using the existing
        seed 42 and 0.2 development fraction. Valid assignments, row order and partition hashes are
        preserved. Duplicate IDs and malformed lineage identities are rejected before feature
        preparation or training. The former <code>aa.guard.train_data</code> module and its unused
        <code> build_training_stream</code> sampler are removed. The active engine retains its own
        checkpointed sampler; manifest preparation retains the fresh-sequence exclusion check.
      </p>
      <p>
        Training commands accept <code>--device</code> and an optional <code>--checkpoints</code> override.
        The feature cache is a JSON file for sweeps and a directory for LOCO folds. Checkpoints bind configuration,
        data, and feature identities; legacy unbound checkpoints are refused. Choose a new checkpoint directory
        for a new run. Research artifacts are not migrated automatically.
      </p>

      <H2 id="feature-cache">Prepare features before tensor assembly</H2>
      <CodeBlock lang="python" code={`from alignair.applicability.profiles import load_bank
from alignair.applicability.feature_cache import precompute_features, sequence_key

bank = load_bank("study/profile-bank")
records = [{"aa": "QVQLVQSGAEVKKPGASVKVSCKASGYTF"}]
cache = precompute_features(
    records, bank, bank_checksum=bank.checksum(),
    cache_path="study/features.json", mask_groups=frozenset(),
)
vector = cache[sequence_key(records[0]["aa"])]`} />
      <p>
        Feature preparation lives in <code>alignair.applicability.feature_cache</code>, separate from
        training targets. It verifies the supplied bank checksum before reuse or extraction, hashes
        exact sequence text, and computes each missing unique sequence once. It does not strip,
        change case or truncate inputs. All requested records are checked before extraction starts.
        Cached vectors require the current feature width and finite numeric values; strings and
        booleans are rejected. Errors include the cache path, record index or sequence digest where applicable.
      </p>
      <p>
        The returned <code>FeatureCache</code> is a read-only mapping retaining the bank checksum,
        ordered feature names and masked profile groups. Indexing returns a list copy;
        <code> dict(cache)</code> gives a JSON-compatible vector mapping. Training and tensor assembly
        reject a bound cache from another bank or fold. Plain in-memory mappings remain supported
        through explicit binding and vector validation; the caller owns their declared identities.
        These checks establish compatibility and shape, not authenticity of feature values.
      </p>
      <p>
        Valid version-1 cache files remain reusable without rewriting or recomputing their features.
        Legacy unbound files and mismatched bank/order/mask identities are rebuilt. Malformed files
        and invalid vectors under a matching identity are refused without replacement. Extensions
        publish atomically after all extraction succeeds. Cache files and checkpoints retain their
        existing logical schemas; vector-only checkpoint hashes preserve native integer/float encodings.
      </p>
      <p>
        Import <code>precompute_features</code> from the cache module; its former
        <code> training.data</code> location is removed. <code>training.data.build_tensors</code>
        consumes prepared features and no longer accepts a <code>bank</code> argument or performs
        HMM searches. Prepare missing sequences first; a missing feature raises
        <code> DataContractError</code>. Tensor labels, supervision masks and train-only normalization
        retain their existing behavior. The sampling policy below defines training-pool eligibility
        and batch allocation.
      </p>
      <H2 id="sampling">Allocate complete training batches</H2>
      <CodeBlock lang="python" code={`from alignair.applicability.training.sampling import GuardBatchSampler

# Pools contain prepared records with aa and group annotations.
sampler = GuardBatchSampler(
    training_records, outlier_records,
    batch_size=32, oe_cap_fraction=0.15, seed=7,
)
main_records, outliers = sampler.draw()
assert len(main_records) + len(outliers) == 32
saved_sampling_state = sampler.state_dict()`} />
      <p>
        Policy <code>alignair.guard.balanced_sampling.v2</code> reserves the integer OE quota
        within its declared cap, then fills every remaining batch slot from active positive,
        biological-negative and structural roles. Role counts differ by at most one. Extra slots
        rotate across roles after each batch; sampling within each pool uses replacement and a
        local RNG seeded with <code>seed + 1000</code>. Main records retain role order and OE
        records form a separate suffix during tensor assembly.
      </p>
      <p>
        Batch size must be a positive integer, and the finite OE cap lies in [0, 0.5). An empty
        OE pool reserves zero outliers. The batch must fit every active role after reserving its
        quota. Unknown training groups, empty sequence text and overlength train/OE records are
        rejected before model allocation; this prevents tensor filtering from changing batch size
        or OE share. The recognized groups are positive_real/positive_sim;
        vhh_nonhuman/igsf/igk_igl/tcr; and structural.
      </p>
      <p>
        The sampler stores both RNG state and the number of batches drawn. Resume verifies the
        allocation plan and requires the batch counter to match checkpoint progress. This preserves
        remainder rotation across interruptions. Malformed state raises a contextual
        <code> TrainingError</code> without partially changing the sampler.
      </p>
      <p>
        New manifests declare <code>sampling_policy</code>; checkpoints and run results record the
        complete <code>sampling_plan</code>. Training contracts use schema version 2 and bind that
        plan. Freeze new manifests to new paths and use new checkpoint directories for corrected
        runs. Historical unversioned programs and prior-policy checkpoints are refused for execution
        or reuse. LOCO seed preparation also verifies that successful results belong to its declared
        training manifest and current sampling policy. Historical artifacts are preserved.
      </p>
      <H2 id="checkpoints">Guard checkpoint compatibility</H2>
      <p>
        <code>checkpoint.load_checkpoint</code> requires the expected <code>training_contract</code>
        and current <code>CheckpointProvenance</code>. Training uses it both to resume interrupted
        optimization and to reuse completed results. Before restoring state or recording success, it
        checks the run hash and the five runtime identities: tokenizer, profile-bank checksum, ordered
        feature names, class-map hash and partition hash. Missing or changed identities raise a contextual
        integrity error with the checkpoint path and mismatched fields. Malformed expected identities
        raise <code>ConfigurationError</code> before deserialization.
      </p>
      <p>
        Seed, manifest, loss settings and software versions remain in provenance for auditing. The run
        hash independently binds settings and data; software versions are not strict runtime pins.
        Version discovery reads installed package metadata without importing pyhmmer or ANARCII;
        unavailable distributions are recorded as <code>n/a</code>.
        Model and optimizer state shapes are checked by their PyTorch loaders. These APIs use trusted
        local pickle files and do not authenticate their origin. Do not load untrusted checkpoints.
      </p>
      <p>
        Both periodic and final saves use <code>alignair.io.atomic</code>, preserving the previous
        checkpoint if serialization or replacement fails. Periodic checkpoints now include provenance;
        older periodic files without it require explicit migration using verified identities or a new
        run. Existing completed checkpoints remain reusable when their run and runtime bindings match.
        Read/write failures retain their causes in the shared error hierarchy, and CUDA allocation
        failures keep their original exception type. Publication is atomic per file, without a whole-run
        transaction or a power-loss guarantee for directory entries.
      </p>
      <p>
        Guard checkpoints use the shared <code>train.random_state</code> implementation for process
        generators and retain a separate <code>sampling_rng</code> state for choosing training records
        and rotating remainder slots.
        Resume requires Python, NumPy and PyTorch CPU states; missing or malformed state raises
        <code> TrainingError</code> with checkpoint and run context. The sampler is checked before
        model or process-state restoration. CPU guard runs exclude CUDA state on every host and
        preserve the explicit null <code>rng.cuda</code> checkpoint field. CUDA runs capture and
        restore all CUDA generators. The sampler’s versioned state includes its allocation plan,
        local RNG and batch counter.
      </p>

      <H2 id="reports">Reports and compute accounting</H2>
      <CodeBlock lang="python" code={`from alignair.applicability.training.budget import record, total

record("C1.s0", wall_s=120.0, gpu_h=0.0, params=45000,
       path="study/results/budget.json")
totals = total(path="study/results/budget.json")
assert totals["n_runs"] >= 1`} />
      <p>
        Accounting stores one entry per run ID. Recording the same ID replaces its totals, so
        reusing a completed checkpoint does not double-count the run. Other runs and additional
        ledger metadata are preserved. Wall seconds and GPU hours must be finite nonnegative
        numbers; parameter counts must be nonnegative integers. Strings and booleans are rejected.
        A missing ledger has zero totals and is not created by <code>total</code>.
      </p>
      <p>
        Both reads and updates validate existing ledgers. Malformed JSON, invalid run records or
        nonfinite values raise <code>ArtifactIntegrityError</code> without replacing the file.
        Invalid new entries raise <code>InputError</code>; read/write failures raise
        <code> OperationIOError</code> with path and stage context. Updates assume one serialized
        writer: atomic replacement prevents partial files, but does not lock concurrent ledger updates.
      </p>
      <DocTable head={["Artifact", "Publication"]} rows={[
        [<code>budget.json</code>, "Updated after a completed run or verified checkpoint reuse."],
        [<><code>stage4_runs.json</code> / <code>loco_runs.json</code></>, "Complete progress snapshot after each returned run result, including engineering failures."],
        [<><code>STAGE4_RESULT.md</code> / <code>STAGE5_LOCO_TRAINING.md</code></>, "Summary after the study loop completes; does not choose eligibility or thresholds."],
        [<code>&lt;run-tag&gt;.failure.json</code>, "Structured diagnostic for a training failure; no success accounting is recorded."],
      ]} />
      <p>
        Reports and ledgers use the shared sibling-file, fsync and atomic-replacement lifecycle.
        JSON reports use <code>alignair.io.atomic.atomic_write_json</code>, which encodes before
        writing and rejects NaN, infinity and unsupported objects. A failed
        replacement keeps the previous file and saved checkpoints. If publishing a failure diagnostic
        also fails, the resulting error retains the original diagnostic in
        <code> context.training_error</code> and chains the publication cause. Files are published
        individually; a study’s files are not a single transaction.
      </p>
      <H2 id="diagnostics">Diagnostics and release checks</H2>
      <p>
        Commands support <code>--debug</code> and <code>--error-format json</code>. For training and OAS commands,
        put these options before the subcommand. See <DocLink to="errors">Errors and diagnostics API</DocLink>
        for diagnostic fields and exception handling.
      </p>
      <p>
        Training smoke coverage exercises the package engine in <code>tests/alignair/applicability/training</code>.
        The <DocLink to="testing">testing guide</DocLink> covers selection and research dependencies.
        The separate <code>tests/packaging/release_smoke.sh</code> runs the public CLI’s version, environment check,
        tiny demo, AIRR validation, and comparison against an installed wheel or Docker image. It uses the
        same check in CI and the release workflow; it does not train research candidates or publish artifacts.
      </p>
    </>
  ),
};

export const aaResearchPages: DocPage[] = [aaGuardResearch];

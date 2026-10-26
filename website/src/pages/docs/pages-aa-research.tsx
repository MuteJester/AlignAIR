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
      <DocTable head={["Module under alignair.aa.guard", "Responsibility"]} rows={[
        [<code>data_sources</code>, "Source-specific collection, exclusions, verified download caches, and optional independent domain numbering."],
        [<code>cohort.run_gate</code>, "Allocation, structural-negative generation, post-allocation deduplication, and the cohort audit."],
        [<code>disjoint</code>, "Reproduce the historical pre-split corpus digest and audit fresh-record overlaps without opening the sealed partition."],
        [<code>profiles.alignment</code>, "Germline allele selection, translation, protein deduplication, and alignment."],
        [<code>profiles.builder.build_profile_bank</code>, "Build a complete HMM bank from checksum-pinned local sources."],
        [<code>training.manifests</code>, "Bind explicit study settings to data, profile, partition, and fold identities."],
        [<code>training.studies</code>, "Run seed sweeps and leave-one-category-out training."],
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
      <CodeBlock code={`python -m alignair.aa.guard.data_sources.oas --help
python -m alignair.aa.guard.data_sources.genairr --help
python -m alignair.aa.guard.data_sources.igsf --help
python -m alignair.aa.guard.data_sources.sabdab --help
python -m alignair.aa.guard.data_sources.tenx_tcr --help`} />
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
      <CodeBlock code={`python -m alignair.aa.guard.cohort \\
  --store study/sources --census-dir study/census \\
  --fingerprint inputs/split_fingerprint.json --out study/cohort

python -m alignair.aa.guard.disjoint fresh.jsonl \\
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
      <CodeBlock code={`python -m alignair.aa.guard.profiles --sources sources.json --out study/profile-bank`} />
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
        the builder does not download replacement inputs. Existing imports from <code>alignair.aa.guard.profiles</code> remain available.
      </p>

      <H2 id="training">Prepare and run training studies</H2>
      <CodeBlock code={`python -m alignair.aa.guard.training freeze-sweep --help
python -m alignair.aa.guard.training freeze-loco --help
python -m alignair.aa.guard.training sweep --help
python -m alignair.aa.guard.training loco --help`} />
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
        Training commands accept <code>--device</code> and an optional <code>--checkpoints</code> override.
        The feature cache is a JSON file for sweeps and a directory for LOCO folds. Checkpoints bind configuration,
        data, and feature identities; legacy unbound checkpoints are refused. Choose a new checkpoint directory
        for a new run. Research artifacts are not migrated automatically.
      </p>

      <H2 id="diagnostics">Diagnostics and release checks</H2>
      <p>
        Commands support <code>--debug</code> and <code>--error-format json</code>. For training and OAS commands,
        put these options before the subcommand. See <DocLink to="errors">Errors and diagnostics API</DocLink>
        for diagnostic fields and exception handling.
      </p>
      <p>
        Training smoke coverage exercises the package engine in <code>tests/alignair/aa/guard/training</code>.
        The <DocLink to="testing">testing guide</DocLink> covers selection and research dependencies.
        The separate <code>tests/packaging/release_smoke.sh</code> runs the public CLI’s version, environment check,
        tiny demo, AIRR validation, and comparison against an installed wheel or Docker image. It uses the
        same check in CI and the release workflow; it does not train research candidates or publish artifacts.
      </p>
    </>
  ),
};

export const aaResearchPages: DocPage[] = [aaGuardResearch];

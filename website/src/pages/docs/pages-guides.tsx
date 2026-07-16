import { CodeBlock, Callout } from "../../components/ui";
import { DocTable, DocLink, type DocPage } from "./doc-kit";

const models: DocPage = {
  slug: "models",
  title: "Pretrained models",
  section: "Guides",
  lead: "The public model hub, how --model resolves, and what each model is.",
  body: () => (
    <>
      <p>
        AlignAIR distributes pretrained models from a public hub,{" "}
        <a href="https://huggingface.co/AlignAIR/AlignAIR-pretrained" target="_blank" rel="noreferrer" className="text-brand-600 underline">
          AlignAIR/AlignAIR-pretrained
        </a>
        . You never download them by hand - pass a model id to <code>--model</code> and the CLI fetches, verifies, and
        caches it on first use. No login required.
      </p>

      <h2>Authoritative model catalog</h2>
      <p>
        The CLI catalog is the live, authoritative list of available models. New models or loci can be added to the hub and discovered instantly without requiring a new AlignAIR package release.
      </p>
      <ul className="list-disc pl-6 space-y-1 my-3">
        <li>Run <code>alignair models list</code> to view the currently available live catalog and check which models are cached locally.</li>
        <li>Run <code>alignair models update</code> to update already installed model checkpoints to their latest version.</li>
        <li>For production pipelines, always pin an exact version (e.g. <code>alignair-igh-human@1.0.0</code>) to guarantee reproducibility.</li>
      </ul>

      <h2>Pretrained model registry</h2>
      <div style={{ margin: "20px 0", overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px", lineHeight: "1.6" }}>
          <thead>
            <tr style={{ background: "#f7f6fb", borderBottom: "2px solid #eae9f1" }}>
              <th style={{ textAlign: "left", padding: "10px 12px", fontWeight: 600 }}>Model ID</th>
              <th style={{ textAlign: "left", padding: "10px 12px", fontWeight: 600 }}>Locus Coverage</th>
              <th style={{ textAlign: "left", padding: "10px 12px", fontWeight: 600 }}>GenAIRR DataConfig</th>
              <th style={{ textAlign: "left", padding: "10px 12px", fontWeight: 600 }}>Upstream Alleles</th>
              <th style={{ textAlign: "left", padding: "10px 12px", fontWeight: 600 }}>Version</th>
              <th style={{ textAlign: "left", padding: "10px 12px", fontWeight: 600 }}>Model created (UTC)</th>
              <th style={{ textAlign: "left", padding: "10px 12px", fontWeight: 600 }}>Alleles (V/D/J)</th>
              <th style={{ textAlign: "left", padding: "10px 12px", fontWeight: 600 }}>Allele-order fingerprint prefix*</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: "1px solid #f0eff5" }}>
              <td style={{ padding: "10px 12px" }}><code>alignair-igh-human</code></td>
              <td style={{ padding: "10px 12px" }}>Human IGH (Heavy)</td>
              <td style={{ padding: "10px 12px" }}><code>HUMAN_IGH_OGRDB</code></td>
              <td style={{ padding: "10px 12px" }}><a href="https://ogrdb.airr-community.org/" target="_blank" rel="noreferrer" className="text-brand-600 underline">OGRDB</a></td>
              <td style={{ padding: "10px 12px" }}>1.0.0</td>
              <td style={{ padding: "10px 12px" }}>2026-07-13</td>
              <td style={{ padding: "10px 12px" }}>198 / 33 / 7</td>
              <td style={{ padding: "10px 12px" }}><code title="d615b59babba6902cb876ed415809df8e565682511dc9ab6032d2fc294dd582a">d615b59babba…</code></td>
            </tr>
            <tr style={{ borderBottom: "1px solid #f0eff5" }}>
              <td style={{ padding: "10px 12px" }}><code>alignair-igkl-human</code></td>
              <td style={{ padding: "10px 12px" }}>Human IGK + IGL (Light)</td>
              <td style={{ padding: "10px 12px" }}><code>HUMAN_IGK_OGRDB</code> + <code>HUMAN_IGL_OGRDB</code></td>
              <td style={{ padding: "10px 12px" }}><a href="https://ogrdb.airr-community.org/" target="_blank" rel="noreferrer" className="text-brand-600 underline">OGRDB</a></td>
              <td style={{ padding: "10px 12px" }}>1.0.0</td>
              <td style={{ padding: "10px 12px" }}>2026-07-13</td>
              <td style={{ padding: "10px 12px" }}>349 / 0 / 18</td>
              <td style={{ padding: "10px 12px" }}><code title="d32cd6fe79114117a504def1321ee831d60c95ef585e1127358d6be36a826463">d32cd6fe7911…</code></td>
            </tr>
            <tr style={{ borderBottom: "1px solid #f0eff5" }}>
              <td style={{ padding: "10px 12px" }}><code>alignair-tcrb-human</code></td>
              <td style={{ padding: "10px 12px" }}>Human TRB (TCR Beta)</td>
              <td style={{ padding: "10px 12px" }}><code>HUMAN_TCRB_IMGT</code></td>
              <td style={{ padding: "10px 12px" }}><a href="https://www.imgt.org/" target="_blank" rel="noreferrer" className="text-brand-600 underline">IMGT</a></td>
              <td style={{ padding: "10px 12px" }}>1.0.0</td>
              <td style={{ padding: "10px 12px" }}>2026-07-13</td>
              <td style={{ padding: "10px 12px" }}>98 / 3 / 16</td>
              <td style={{ padding: "10px 12px" }}><code title="657d771a8952784d913936d5eef1020ac0395528266790cae23ec928b935c359">657d771a8952…</code></td>
            </tr>
          </tbody>
        </table>
      </div>
      <p style={{ fontSize: "12px", color: "#6f6d85", marginTop: "-10px" }}>
        * Note: The allele-order fingerprint (<code>allele_order_sha256</code>) is the SHA-256 hash of the list of ordered alleles embedded in the model card to prevent reference drift; it is not the hash of the model checkpoint file or reference FASTA. Tooltips display full hash values.
      </p>

      <Callout kind="note" title="TCR Pretrained Coverage Limitation">
        Pretrained TCR support is strictly limited to <strong>human TRB (TCR Beta)</strong>. Other TCR loci (like TRA, TRG, TRD) or non-human species are not supported by the pretrained models, but can be fully trained as custom models using appropriate germline reference FASTAs.
      </Callout>

      <h2>Using a model</h2>
      <CodeBlock code={`alignair predict --input reads.fasta --out out.tsv --model alignair-igh-human\n# pin an exact version for reproducibility\nalignair predict --input reads.fasta --out out.tsv --model alignair-igh-human@1.0.0`} />
      <p>
        On first use the model is downloaded, its SHA-256 is checked against the catalog, and it is cached locally.{" "}
        <code>--model</code> also accepts a local <code>.alignair</code> path or an <code>org/name</code> Hugging Face
        repo id.
      </p>

      <h2>Model cards &amp; scientific limitations</h2>
      <p>All three models share this configuration:</p>
      <ul>
        <li>Convolutional encoder with per-gene segmentation + classification heads, plus orientation, mutation-rate, indel, and productivity heads.</li>
        <li>Maximum input length 576 nt (longer reads are cropped and flagged <code>length_cropped</code>).</li>
        <li>Intended input: single-end reads, full-length or fragments, any orientation. Not paired-read assembly, not BAM/CRAM.</li>
        <li>Fixed-reference classifier; constrain with <code>--genotype</code>, add alleles by training.</li>
        <li>Post-hoc temperature-scaling calibration (below). Minimum AlignAIR 2.0.0; <code>.alignair</code> v1 (pickle-free).</li>
        <li>License GPL-3.0-or-later. Citation: Konstantinovsky et al., Nucleic Acids Research 2025, gkaf651.</li>
      </ul>
      <h3>Per-model specifications &amp; limitations</h3>
      <DocTable
        head={["Model", "Reference Specs", "Known Scientific Limitations"]}
        rows={[
          [<code>alignair-igh-human</code>, "GenAIRR OGRDB, V198 / D33 / J7", "On full-length heavy-SHM V, accuracy is comparable to IgBLAST rather than higher; junctions can jitter ~1-2 nt. D-allele calling is inherently ambiguous due to chew-off deletions."],
          [<code>alignair-igkl-human</code>, "GenAIRR OGRDB, V349 / J18 (no D)", <>Multi-locus (IGK + IGL). <code>d_call</code> and <code>np2</code> empty by design; each read attributed to one locus. Designed for light chain alignment.</>],
          [<code>alignair-tcrb-human</code>, "GenAIRR IMGT, V98 / D3 / J16", <>SHM set to zero (TCR does not hypermutate). Short TRB D often yields a multi-member <code>d_call_set</code> or a blank <code>d_call</code>. Limited to TRB locus; other TCR chains are not supported by this checkpoint.</>],
        ]}
      />
      <p>
        Training simulation for all: GenAIRR-native, curriculum-ramped (SHM, indels, sequencing error, ambiguous bases,
        one-sided and both-ends fragmentation, orientation augmentation). Exact fingerprints and calibration temperatures
        are in <code>alignair models info &lt;id&gt;</code>.
      </p>
      <Callout kind="note" title="Held-out validation metrics">
        Per-model held-out accuracy is not yet published in the registry (<code>metrics</code> is null). Evaluate a model
        yourself with <code>alignair benchmark --model &lt;id&gt; --n 200 --out benchmark.json</code> (see{" "}
        <DocLink to="benchmarks">Benchmarks</DocLink>).
      </Callout>

      <h2>Confidence calibration</h2>
      <p>
        Each pretrained model carries post-hoc temperature scaling on the allele heads: a per-gene temperature fitted on
        held-out data, embedded in the model card and hash-verified. It is applied automatically; there is no flag.
        Temperature scaling is monotonic, so it never reorders the alleles and never changes which one is called - it
        rescales the reported probability so it better matches observed accuracy. It <em>can</em> change{" "}
        <code>*_call_set</code> membership, though: the set rule is an absolute <code>p &gt;= 0.5</code> cut rather than
        a rank cut, and rescaling moves probabilities across that line. Held-out calibration quality metrics are not yet
        published; the <code>*_set_confidence</code> column stays blank until that step.
      </p>

      <h2>Managing the cache</h2>
      <CodeBlock code={`alignair models get alignair-tcrb-human      # pre-download (id or id@version)\nalignair models info alignair-igh-human      # model card (calibration, fingerprints)\nalignair models path alignair-igh-human      # cached file path\nalignair models verify                       # re-hash installed models\nalignair models update                       # update to the latest version\nalignair models prune                        # remove old cached versions`} />

      <h2>Offline and private registries</h2>
      <ul>
        <li><code>--offline</code> uses only what is already cached (<code>ALIGNAIR_NO_NETWORK=1</code> also disables the update check).</li>
        <li>Point at another registry with <code>--registry</code>, <code>$ALIGNAIR_REGISTRY</code>, or a config file - a Hugging Face repo, an <code>https://</code> mirror, or a local <code>file://</code> directory. This is how a lab hosts private models.</li>
      </ul>

      <h2>What a model file contains</h2>
      <p>
        Each <code>.alignair</code> is self-contained and safe to load: weights plus the germline reference (sequences +
        anchors), fingerprinted so it cannot drift from the weights, and it loads without executing any pickle. Every
        prediction run also writes a <code>&lt;out&gt;.run.json</code> provenance sidecar. Publishing to the hub is a
        maintainer-only, token-gated step (see <DocLink to="publishing">Publishing models</DocLink>).
      </p>
    </>
  ),
};

const training: DocPage = {
  slug: "training",
  title: "Training a custom model",
  section: "Guides",
  lead: "Train a model for a reference AlignAIR does not ship, and judge whether to trust it.",
  body: () => (
    <>
      <p>
        Train when you need a reference AlignAIR does not already ship: a different species, a custom germline set, or a
        locus none of the pretrained models cover. Training produces a new fixed-reference model whose embedded reference
        is the exact callable set. The output is a self-contained, pickle-free bundle.
      </p>

      <h2>Two ways to specify the reference</h2>
      <p>Pick exactly one mode. A built-in GenAIRR dataconfig:</p>
      <CodeBlock code={`alignair train --dataconfig HUMAN_IGH_OGRDB --out runs/igh --preset desktop`} />
      <p>
        List names with <code>alignair reference list</code>. Several dataconfigs train one multi-locus model. Or your
        own germline FASTAs:
      </p>
      <CodeBlock code={`alignair train --v-fasta v.fasta --j-fasta j.fasta --d-fasta d.fasta \\\n  --chain-type BCR_HEAVY --out runs/my_ref --preset desktop`} />

      <h3>Chain types and the D segment</h3>
      <p>
        <code>--chain-type</code> is one of <code>BCR_HEAVY</code>, <code>BCR_LIGHT_KAPPA</code>,{" "}
        <code>BCR_LIGHT_LAMBDA</code>, <code>TCR_ALPHA</code>, <code>TCR_BETA</code>, <code>TCR_GAMMA</code>,{" "}
        <code>TCR_DELTA</code>. D-bearing chains (<code>BCR_HEAVY</code>, <code>TCR_BETA</code>, <code>TCR_DELTA</code>)
        require <code>--d-fasta</code>; the others must not be given one.
      </p>

      <h3>Reference FASTA rules</h3>
      <ul>
        <li>Each file is a standard germline FASTA (allele name in the header, nucleotide sequence).</li>
        <li>V and J conserved anchors are discovered during the build; alleles whose anchor cannot be placed still get calls but have no junction (honest absence).</li>
        <li>Unusable (malformed / duplicate) alleles are rejected and counted, recorded in <code>reference_manifest.json</code> with per-gene <code>anchored</code> coverage. Check anchor coverage before committing to a reference.</li>
      </ul>

      <h2>Presets</h2>
      <DocTable
        head={["Preset", "Steps", "Batch", "Validation", "Purpose"]}
        rows={[
          [<code>quick</code>, "300", "16", "every 100", "Software smoke test / CI only"],
          [<code>desktop</code>, "50,000", "64", "every 2,000", "A single workstation GPU"],
          [<code>full</code>, "300,000", "128", "every 5,000", "A production, paper-grade run"],
        ]}
      />
      <Callout kind="warning" title="quick is not a scientific model">
        The <code>quick</code> preset (and <code>alignair demo</code>) train for a few hundred steps only to prove the
        pipeline runs. Their calls are not accurate. Use <code>desktop</code> or <code>full</code> for a model you will
        make calls with.
      </Callout>

      <h2>Preview before training</h2>
      <CodeBlock code={`alignair train --dataconfig HUMAN_IGH_OGRDB --out runs/igh --preset desktop --plan`} />
      <p>
        <code>--plan</code> validates the reference and configuration and prints the plan without training (resolved
        loci, per-gene allele counts, whether the locus has D, max input length, model parameter count, and the step /
        batch schedule). It does not estimate wall time or memory.
      </p>

      <h2>How training data is simulated</h2>
      <p>
        AlignAIR trains on GenAIRR-simulated reads with known ground truth (no post-hoc cropping - read length and every
        corruption come from the simulation). A curriculum ramps difficulty, drawing somatic hypermutation (S5F, ~0.5 to
        15%; set to zero for TCR), indels (0 to ~5), sequencing error (0 to ~2%), ambiguous bases, and read orientation
        (up to ~50% non-forward). An amplicon mix spans full-length, V/framework-anchored, J-anchored, and both-ends
        fragments, plus a heavily-mutated full-length stream.
      </p>

      <h2>What training writes</h2>
      <DocTable
        head={["File", "Contents"]}
        rows={[
          [<code>model.alignair</code>, "The pickle-free model: weights + embedded, fingerprinted reference."],
          [<code>model_card.md</code>, "Loci, per-gene allele counts, training steps, and the validation summary."],
          [<code>reference_manifest.json</code>, "Allele counts and anchor coverage, reference fingerprints, source hashes, tool versions."],
          [<code>validation_report.json</code>, "Per-task metrics on a fixed held-out stream (below)."],
        ]}
      />
      <p>
        The export fails closed: an existing <code>bundle/</code> is not overwritten unless you pass{" "}
        <code>--overwrite</code>. Predict with{" "}
        <code>--model runs/igh/bundle/model.alignair</code>.
      </p>

      <h2>Interpreting validation_report.json</h2>
      <DocTable
        head={["Metric", "Meaning", "Direction"]}
        rows={[
          [<code>v/d/j_allele_top1</code>, "Top call is inside the truth equivalence set (headline accuracy).", "Higher (0-1)"],
          [<code>v/d/j_seg_mae</code>, "Mean absolute error of the segment start and end, in nucleotides.", "Lower"],
          [<code>orientation_acc</code>, "Orientation-head accuracy.", "Higher"],
          [<code>chain_type_acc</code>, "Locus classification accuracy (multi-locus).", "Higher"],
          [<code>mutation_mae</code>, "Error of the SHM-rate estimate.", "Lower"],
          [<code>indel_mae</code>, "Error of the indel-count estimate.", "Lower"],
          [<code>productive_acc</code>, "Productivity-head accuracy.", "Higher"],
        ]}
      />
      <p>
        The best-checkpoint score is the mean of <code>v/d/j_allele_top1</code>. Guidance: V and J top-1-in-set should be
        well above 0.9 on this stream for a <code>desktop</code>/<code>full</code> run; D is expected lower (especially
        short-D loci); segmentation MAE should be single-digit nucleotides; orientation near 1.0.
      </p>
      <Callout kind="note" title="Necessary, not sufficient">
        The held-out stream scores the model on the same GenAIRR simulation family it trained on. It is a strong signal,
        but benchmark on reads representative of your data before trusting a model.
      </Callout>
    </>
  ),
};

export const guidesPages: DocPage[] = [models, training];

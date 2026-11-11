import { CodeBlock, Callout } from "../../components/ui";
import { DocTable, DocLink, type DocPage } from "./doc-kit";

const models: DocPage = {
  slug: "models",
  title: "Pretrained models",
  section: "Using AlignAIR",
  lead: "The public model hub, how --model resolves, and what an AlignAIR 4 model is.",
  body: () => (
    <>
      <p>
        AlignAIR distributes pretrained models from a public hub,{" "}
        <a href="https://huggingface.co/AlignAIR/AlignAIR-pretrained" target="_blank" rel="noreferrer" className="text-brand-600 underline">
          AlignAIR/AlignAIR-pretrained
        </a>
        . You never download them by hand - pass a model id or a Hugging Face repo to <code>--model</code> and AlignAIR
        fetches and caches it on first use. A public model needs no login.
      </p>

      <h2>AlignAIR 4 models: one per locus</h2>
      {/* PENDING-V4-PUBLISH: none of the alignair4-* ids is published yet. */}
      <p>
        AlignAIR 4 pretrained models come one per locus, all in the shared catalog, published in this order: IGH
        first, then IGK, IGL, TRA, TRB, TRD and TRG. None is published yet; until then, any AlignAIR 4{" "}
        <code>.alignair</code> path works in its place.
      </p>
      <DocTable
        head={["Locus", "Catalog id"]}
        rows={[
          ["IGH", <code>alignair4-igh-human</code>],
          ["IGK", <code>alignair4-igk-human</code>],
          ["IGL", <code>alignair4-igl-human</code>],
          ["TRA", <code>alignair4-tra-human</code>],
          ["TRB", <code>alignair4-trb-human</code>],
          ["TRD", <code>alignair4-trd-human</code>],
          ["TRG", <code>alignair4-trg-human</code>],
        ]}
      />
      <Callout kind="warning" title="The catalog's current entries are AlignAIR 3 models">
        The hub catalog currently lists <code>alignair-igh-human</code>, <code>alignair-igkl-human</code> and{" "}
        <code>alignair-tcrb-human</code>, all at version 1.0.0 and all AlignAIR 3 models (architecture{" "}
        <code>conv-v1</code>). AlignAIR 4 refuses them: <em>this is an AlignAIR 3 model (architecture 'conv-v1');
        AlignAIR 4 reads chip models only - use alignair&lt;4 to run it</em>.
      </Callout>

      <h2>The live catalog</h2>
      <p>
        The catalog is fetched live from the hub, so new models can be added without a new AlignAIR release.
      </p>
      <ul className="list-disc pl-6 space-y-1 my-3">
        <li>Run <code>alignair models list</code> to view the catalog and see which models are cached locally.</li>
        <li>Run <code>alignair models update</code> to update installed models to their latest version.</li>
        <li>For production pipelines, pin an exact version (<code>id@version</code>) or a Hugging Face revision to guarantee reproducibility.</li>
      </ul>

      <h2>Using a model</h2>
      <CodeBlock code={`alignair predict --input reads.fasta --out out.tsv --model alignair4-igh-human\n# pin an exact catalog version for reproducibility\nalignair predict --input reads.fasta --out out.tsv --model alignair4-igh-human@<version>\n# or a Hugging Face repo, pinned to a branch, tag or commit\nalignair predict --input reads.fasta --out out.tsv --model hf://<org>/<repo> --revision <commit>`} />
      <CodeBlock lang="python" code={`from alignair import Aligner

aligner = Aligner.from_pretrained("hf://<org>/<repo>", revision="<commit>", device="auto")
aligner.predict(["CAGGTGCAGCTG..."]).write_airr("out.tsv")`} />
      <p>
        A catalog model is downloaded on first use, its SHA-256 checked against the catalog, and cached locally. A
        Hugging Face repo (<code>hf://org/repo</code> or <code>org/repo</code>) is pulled with{" "}
        <code>huggingface_hub</code> (in the <code>[cli]</code> and <code>[hub]</code> extras) into the Hugging Face cache;
        <code> --hf-token</code> (or <code>$HF_TOKEN</code>) opens a private repo, and the default file is{" "}
        <code>model.alignair</code>. <code>--model</code> also accepts a local <code>.alignair</code> path.
      </p>

      <h2>Model cards &amp; scientific limitations</h2>
      <p>An AlignAIR 4 model file:</p>
      <ul>
        <li>holds the <code>chip-v1</code> network: an orientation correction, a dilated convolutional trunk, a region parser, V/J anchor heads, a mutation reader, and per-segment scans of the bound reference's alleles;</li>
        <li>reads at most 575 nt (a 576-position window less one reserved slot); longer reads are cropped and flagged <code>length_cropped</code>;</li>
        <li>takes single-end reads, full-length or fragments, in any orientation - not paired-read assembly, not BAM/CRAM;</li>
        <li>embeds its training reference as the default bank; restrict it with <code>--genotype</code>, or bind another with <code>--reference</code> (see the <DocLink to="model-contract">model contract</DocLink>);</li>
        <li>is <code>.alignair</code> format v1 and loads without unpickling anything;</li>
        <li>is licensed GPL-3.0-or-later. Citation: Konstantinovsky et al., Nucleic Acids Research 2025, gkaf651.</li>
      </ul>
      <p>
        <code>alignair info model.alignair</code> prints a file's card (format, creating version, allele counts,
        parameters, training steps, reference, sections); <code>alignair models info &lt;id&gt;</code> does the same for a
        catalog id. A bundle from <code>alignair train</code> also has a <code>model_card.md</code> and a{" "}
        <code>reference_manifest.json</code> beside the model file.
      </p>
      <Callout kind="note" title="Held-out validation metrics">
        {/* PENDING-V4-NUMBERS: per-model held-out accuracy for the v4 models. */}
        Per-model accuracy for the AlignAIR 4 models is pending their release and a fresh head-to-head. Evaluate a model
        yourself with <code>alignair benchmark --model &lt;model&gt; --n 200 --format json --out benchmark.json</code>{" "}
        (see <DocLink to="benchmarks">Benchmarks</DocLink>).
      </Callout>

      <h2>Managing the cache</h2>
      <CodeBlock code={`alignair models get alignair4-igh-human      # pre-download (id or id@version)\nalignair models info alignair4-igh-human     # model card (allele counts, fingerprints)\nalignair models path alignair4-igh-human     # cached file path\nalignair models verify                  # re-hash installed models\nalignair models update                  # update to the latest version\nalignair models prune                   # remove old cached versions`} />

      <h2>Offline and private registries</h2>
      <ul>
        <li><code>--offline</code> uses only what is already cached; <code>ALIGNAIR_NO_NETWORK=1</code> turns off the passive update check.</li>
        <li>Point at another registry with <code>--registry</code>, <code>$ALIGNAIR_REGISTRY</code>, or a config file - a Hugging Face repo, an <code>https://</code> mirror, or a local <code>file://</code> directory. This is how a lab hosts private models.</li>
      </ul>

      <h2>What a model file contains</h2>
      <p>
        Each <code>.alignair</code> is self-contained and safe to load: weights plus the germline reference (sequences,
        IMGT gaps and anchors), fingerprinted so it cannot drift from the weights, and it loads without executing any
        pickle. A training checkpoint also holds pickled sections for resuming (<code>dataconfig/0</code>,{" "}
        <code>train_state</code>); loading never decodes them, but share the pickle-free bundle instead (see{" "}
        <DocLink to="training">Training</DocLink>). Every prediction run also writes a <code>&lt;out&gt;.run.json</code>{" "}
        provenance sidecar. Publishing to the hub is a maintainer-only, token-gated step (see{" "}
        <DocLink to="publishing">Publishing models</DocLink>).
      </p>
    </>
  ),
};

const training: DocPage = {
  slug: "training",
  title: "Training a custom model",
  section: "Training",
  lead: "Train a model for a reference AlignAIR does not ship, and judge whether to trust it.",
  body: () => (
    <>
      <p>
        Train when you need a locus or species no available model covers. For novel alleles or a curated germline set of
        a locus a model already covers, bind it at predict time with <code>--reference</code> instead (see the{" "}
        <DocLink to="model-contract">model contract</DocLink>). Training produces a model whose embedded reference is its
        default bank, and exports a self-contained, pickle-free bundle.
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
        <li>GenAIRR's cartridge builder annotates the V and J conserved anchors; <code>--plan</code> prints its reference-build warnings (e.g. alleles it could not annotate). IMGT-gapped V sequences give it the most to work with.</li>
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
        batch schedule). It does not estimate wall time or memory. Training refuses up front, before any step, when{" "}
        <code>--out</code> already holds a <code>bundle/</code> and <code>--overwrite</code> is not given.
      </p>

      <h2>How training data is simulated</h2>
      <p>
        AlignAIR trains on GenAIRR-simulated reads with known ground truth (no post-hoc cropping - read length and every
        corruption come from the simulation). A curriculum ramps difficulty with its progress: somatic hypermutation
        (S5F, a rate from 0.5% to 15%; none for TCR loci), polymerase indels (on 5% of reads rising to 25%, one to three
        each), sequencing error (up to about 1.5%), up to five ambiguous bases, and up to 40 nt lost from each end. Each
        read is presented in one of four orientations (forward, reverse complement, complement, reverse), drawn
        uniformly. The default mix draws from three curriculum stages, three amplicon shapes (V-anchored, J-anchored and
        both-ends fragments) and a heavily mutated stream (rate 0.25). By default each read also gets library flanks
        around its V(D)J - a 5' leader, a masked primer, a 3' constant region - so the model does not learn that V
        starts at the read's first base.
      </p>

      <h2>What training writes</h2>
      <DocTable
        head={["File", "Contents"]}
        rows={[
          [<code>runs/igh/model.alignair</code>, "The last resumable checkpoint: weights, reference, and pickled training sections (dataconfig/0, train_state). For resuming, not for sharing."],
          [<code>runs/igh/model.step&lt;N&gt;.alignair</code>, "Periodic resumable checkpoints."],
          [<code>runs/igh/model.best.alignair</code>, "The checkpoint with the best validation score, when validation ran; the bundle is exported from it."],
          [<code>runs/igh/bundle/model.alignair</code>, "The pickle-free model to share and predict with: weights + embedded, fingerprinted reference."],
          [<code>bundle/model_card.md</code>, "Loci, per-gene allele counts, training steps, the SHM cap applied per locus, and the validation summary."],
          [<code>bundle/reference_manifest.json</code>, "Allele names and anchor coverage, reference fingerprints, source hashes, tool versions."],
          [<code>bundle/validation_report.json</code>, "Per-task metrics on a fixed held-out stream (below)."],
        ]}
      />
      <p>
        The export fails closed: an existing <code>bundle/</code> is not overwritten unless you pass{" "}
        <code>--overwrite</code>. Predict with{" "}
        <code>--model runs/igh/bundle/model.alignair</code>.
      </p>

      <h2>Sharing a trained model</h2>
      <p>
        Share the bundle, never a checkpoint. Training from Python with <code>run_training</code> writes the
        checkpoints but no bundle; <code>alignair export</code> writes one from any checkpoint, with the model id and
        version in its card:
      </p>
      <CodeBlock code={`alignair export runs/igh/model.alignair runs/igh/release \\\n  --model-id alignair4-igh-human --model-version 1.0.0 --description "Human IGH"\n# -> runs/igh/release/{model.alignair, model_card.md, reference_manifest.json, validation_report.json}`} />
      <p>
        It refuses an existing directory unless <code>--overwrite</code>; <code>--no-validate</code> skips the
        validation report (the card then marks its metrics as pending). A built-in dataconfig is recovered by its name;
        a custom one is read from the checkpoint's own pickled section, so export only checkpoints you trust. In Python
        it is <code>alignair.train.build.export_bundle(checkpoint, dataconfigs, out_dir, model_id=..., model_version=...)</code>,
        which <code>alignair train</code> calls at the end. The card names the model, its reference and version, allele
        counts, training steps, intended use, and the genotype and reference-swap usage.
      </p>
      <p>
        <code>Aligner.save_pretrained(directory)</code> copies a pickle-free model file, and{" "}
        <code>Aligner.push_to_hub(repo_id)</code> uploads it, with the <code>model_card.md</code> and{" "}
        <code>reference_manifest.json</code> beside it, to a Hugging Face repo of its own (needs{" "}
        <code>huggingface_hub</code> and a write token). Both refuse a checkpoint, with a hint naming{" "}
        <code>alignair export</code>, and an aligner made by <code>bound_to</code> (it is not a file). Models published
        for everyone go into the shared catalog instead: see <DocLink to="publishing">Publishing models</DocLink>.{" "}
        <code>alignair convert</code> converts only lab chip model files (<code>lab-chip-v1</code>,{" "}
        <code>lab-chipw64-v1</code>), not training checkpoints.
      </p>

      <h2>Interpreting validation_report.json</h2>
      <DocTable
        head={["Metric", "Meaning", "Direction"]}
        rows={[
          [<code>v/d/j_allele_top1</code>, "The top-scoring allele is in the read's truth allele set (headline accuracy).", "Higher (0-1)"],
          [<code>scan_v/d/j_top1</code>, "The same, from the bank scan's log-posterior; the best-checkpoint score is the mean of these.", "Higher (0-1)"],
          [<code>v/d/j_seg_mae</code>, "Mean absolute error of the segment start and end, in nucleotides.", "Lower"],
          [<code>parse_v/d/j_start/end_exact</code>, "The parser's boundary is exactly the true one.", "Higher (0-1)"],
          [<code>anchor_v/j_exact</code>, "The anchor head finds the conserved codon's exact start.", "Higher (0-1)"],
          [<code>coord_v/d/j_top1</code>, "The mutation reader's top germline column for a base is the true one.", "Higher (0-1)"],
          [<code>bank_v/d/j_size</code>, "Mean number of bank alleles a read is scored over (the batch's bank).", "Informational"],
          [<code>orientation_acc</code>, "Orientation-head accuracy.", "Higher"],
          [<code>d_presence_acc</code>, "Whether a read has a D, called correctly.", "Higher"],
          [<code>mutation_mae</code>, "Error of the SHM-rate estimate.", "Lower"],
          [<code>indel_mae</code>, "Error of the indel-count estimate.", "Lower"],
          [<code>shm_brier</code>, "Brier score of the reader's per-base mutation state.", "Lower"],
        ]}
      />
      <p>
        The best checkpoint is the one with the highest mean of <code>scan_v/d/j_top1</code> on the validation batches.
        {/* PENDING-V4-NUMBERS: target values for a desktop/full v4 run. */} Target values for a{" "}
        <code>desktop</code> or <code>full</code> AlignAIR 4 run are pending (results pending); compare against a
        reference run of the same preset meanwhile. D is expected lower than V and J, especially on short-D loci.
      </p>
      <Callout kind="note" title="Necessary, not sufficient">
        The held-out stream scores the model on the same GenAIRR simulation family it trained on. It is a strong signal,
        but benchmark on reads representative of your data before trusting a model.
      </Callout>
    </>
  ),
};

export const guidesPages: DocPage[] = [models];
export const trainingPages: DocPage[] = [training];

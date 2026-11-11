import { CodeBlock, Callout } from "../../components/ui";
import { type DocPage } from "./doc-kit";

const publishing: DocPage = {
  slug: "publishing",
  title: "Publishing models",
  section: "Maintainer",
  lead: "Maintainer runbook: how pretrained models reach the public hub. Read-only for most users.",
  body: () => (
    <>
      <p>
        AlignAIR distributes pretrained models through a single public Hugging Face repository,{" "}
        <a href="https://huggingface.co/AlignAIR/AlignAIR-pretrained" target="_blank" rel="noreferrer">
          AlignAIR/AlignAIR-pretrained
        </a>
        . It holds a <code>registry.json</code> catalog plus one pickle-free <code>&lt;id&gt;/&lt;version&gt;.alignair</code>{" "}
        per model. A model can also live in its own Hugging Face repo, loaded as <code>hf://org/repo</code>.
        AlignAIR 4 models come one per locus (IGH first, then IGK, IGL, TRA, TRB, TRD, TRG); their ids and repos are
        not decided, so <code>{"<v4-igh-model>"}</code> and <code>{"<org>/<repo>"}</code> below are placeholders.
      </p>
      <ul>
        <li><strong>Anyone can read/download</strong> - <code>alignair models list</code> and <code>predict --model &lt;id&gt;</code> fetch anonymously.</li>
        <li><strong>Only maintainers can publish</strong> - uploading requires a Hugging Face write token. Nothing in the shipped package can write to the repo.</li>
        <li><strong>The catalog auto-updates with no code change</strong> - the CLI reads <code>registry.json</code> live. Re-upload it plus a new artifact and every user sees the change.</li>
      </ul>

      <Callout kind="warning">
        This page is a maintainer runbook. The <code>alignair models</code> CLI exposes only{" "}
        <code>list/get/path/info/update/verify/prune</code> - there is no publish command in the shipped package; publishing goes through the Python calls below.
      </Callout>

      <h2>1. Start from a pickle-free bundle</h2>
      <p>
        Publish only the pickle-free <code>bundle/model.alignair</code> that <code>alignair train</code> exports (or{" "}
        <code>alignair.train.build.export_bundle</code> writes from a checkpoint) - never a training checkpoint, which
        holds pickled <code>dataconfig/0</code> and <code>train_state</code> sections. Export from an installed
        release: a file written from a source tree records <code>created_by_alignair</code> as <code>0+unknown</code>.
      </p>

      <h2>2. Card it</h2>
      <p>
        The catalog requires the card's <code>model_id</code> and <code>model_version</code> to match its entry.{" "}
        <code>update_card</code> rewrites descriptive header fields only; every section (weights, config, reference) is
        carried over byte for byte, and structural or integrity fields are refused:
      </p>
      <CodeBlock lang="python" code={`from alignair.model_file import update_card

update_card("runs/igh/bundle/model.alignair", "release/model.alignair",
            {"model_id": "<v4-igh-model>", "model_version": "<version>", "locus": "IGH", "species": "Homo sapiens"})`} />
      {/* PENDING-V4-MODEL-ID: the v4 ids and versions are not decided. */}

      <h2>3a. Into the catalog</h2>
      <CodeBlock lang="python" code={`from alignair.registry import publish_local

problems = publish_local("release/model.alignair", "<v4-igh-model>", "<version>", ".private/registry",
                         description="Human IGH aligner (AlignAIR 4)")
assert not problems, problems`} />
      <p>
        <code>publish_local</code> stages the artifact, builds the new <code>registry.json</code> in memory, runs the
        validator on that staged state, and commits both atomically only if it passes; on failure it returns the
        problems and writes nothing. Then upload the registry directory (token-gated):
      </p>
      <CodeBlock code={`hf upload AlignAIR/AlignAIR-pretrained .private/registry . \\\n    --repo-type model --no-private --exclude "*.lock"`} />
      <p>
        <code>--no-private</code> keeps the repo world-readable and creates it on first upload. Then verify as an
        anonymous user:
      </p>
      <CodeBlock code={`alignair models list\nalignair predict --input examples/reads.fasta --out out.tsv --model <v4-igh-model>@<version>`} />

      <h2>3b. Or one Hugging Face repo per model</h2>
      <CodeBlock lang="python" code={`from alignair import Aligner

Aligner.from_pretrained("release/model.alignair").push_to_hub("<org>/<repo>", private=False)
# users: Aligner.from_pretrained("hf://<org>/<repo>", revision="<tag or commit>")`} />
      <p>
        <code>push_to_hub</code> needs <code>huggingface_hub</code> and a write token (<code>token=</code> or{" "}
        <code>$HF_TOKEN</code>); it creates the repo (private unless <code>private=False</code>) and uploads{" "}
        <code>model.alignair</code> only - add the bundle's <code>model_card.md</code> and{" "}
        <code>reference_manifest.json</code> yourself. It refuses a checkpoint, as <code>save_pretrained</code> does.
      </p>

      <h2>What gets validated before anything publishes</h2>
      <p>
        Every version is checked by the registry validator: size + SHA-256 match, the artifact is pickle-free (no
        <code> dataconfig</code>/<code>train_state</code> sections) and carries a safe <code>reference_json</code>, the
        card's <code>model_id</code>/<code>model_version</code> match the registry, and the reference hashes recompute
        from the embedded reference. <code>publish_local</code> commits only if it passes, so a broken artifact never
        reaches the upload step.
      </p>
    </>
  ),
};

export const maintainerPages: DocPage[] = [publishing];

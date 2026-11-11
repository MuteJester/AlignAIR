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
        . It holds a <code>registry.json</code> catalog plus, per model version, a pickle-free{" "}
        <code>&lt;id&gt;/&lt;version&gt;.alignair</code> with its <code>.model_card.md</code> and{" "}
        <code>.reference_manifest.json</code> beside it. Users load a model by id:{" "}
        <code>Aligner.from_pretrained("&lt;id&gt;")</code> or <code>--model &lt;id&gt;</code>.
        AlignAIR 4 models come one per locus (IGH first, then IGK, IGL, TRA, TRB, TRD, TRG), all in this shared
        catalog, as <code>alignair4-&lt;locus&gt;-human</code>; none is published yet.{" "}
        {/* PENDING-V4-PUBLISH: the alignair4-* ids are not published yet. */}
      </p>
      <ul>
        <li><strong>Anyone can read/download</strong> - <code>alignair models list</code> and <code>predict --model &lt;id&gt;</code> fetch anonymously.</li>
        <li><strong>Only maintainers can publish</strong> - uploading requires a Hugging Face write token; no CLI command writes to the repo.</li>
        <li><strong>The catalog auto-updates with no code change</strong> - the CLI reads <code>registry.json</code> live. Re-upload it plus a new artifact and every user sees the change.</li>
      </ul>

      <Callout kind="warning">
        This page is a maintainer runbook. The <code>alignair models</code> CLI exposes only{" "}
        <code>list/get/path/info/update/verify/prune</code> - there is no publish command; publishing goes through the Python calls below.
      </Callout>

      <h2>1. Export the bundle with its id and version</h2>
      {/* PENDING-V4-PUBLISH: alignair4-igh-human is not published yet; its first version is not set. */}
      <CodeBlock code={`alignair export runs/igh/model.alignair release/igh \\\n  --model-id alignair4-igh-human --model-version <version> --description "Human IGH (AlignAIR 4)"`} />
      <p>
        <code>alignair export</code> writes the pickle-free <code>model.alignair</code> (no <code>dataconfig/0</code> or{" "}
        <code>train_state</code> section), <code>model_card.md</code>, <code>reference_manifest.json</code> and{" "}
        <code>validation_report.json</code>, with the id and version in the model file's card, which the catalog checks.
        Never publish a training checkpoint. Export from an installed release: a file written from a source tree records{" "}
        <code>created_by_alignair</code> as <code>0+unknown</code>. A bundle exported without an id or version can be
        carded afterwards with <code>alignair.model_file.update_card</code>, which rewrites descriptive header fields only
        and carries every section over byte for byte.
      </p>

      <h2>2. Add it to a local copy of the catalog</h2>
      <CodeBlock code={`hf download AlignAIR/AlignAIR-pretrained --repo-type model --local-dir .private/registry`} />
      <CodeBlock lang="python" code={`from alignair.registry import publish_bundle

problems = publish_bundle("release/igh", "alignair4-igh-human", "<version>", ".private/registry",
                          description="Human IGH aligner (AlignAIR 4)")
assert not problems, problems`} />
      <p>
        Start from a full copy of the hub's catalog: <code>publish_bundle</code> validates every version the local{" "}
        <code>registry.json</code> lists, and that file replaces the hub's when uploaded. It lays the bundle out as the
        catalog serves it, <code>alignair4-igh-human/&lt;version&gt;.alignair</code> with{" "}
        <code>.model_card.md</code> and <code>.reference_manifest.json</code> beside it, records the card and manifest
        in the version's entry with their SHA-256, and stages, validates and commits the three files and the catalog
        together; on failure it returns the problems and writes nothing. Check it loads before uploading:{" "}
        <code>Aligner.from_pretrained("alignair4-igh-human", registry=["file://.private/registry"])</code>.
      </p>

      <h2>3. Upload the version (token-gated)</h2>
      <CodeBlock lang="python" code={`from alignair.registry import upload_version

upload_version(".private/registry", "alignair4-igh-human", "<version>")   # HF_TOKEN: a write token
# one commit to AlignAIR/AlignAIR-pretrained: the .alignair, its .model_card.md and
# .reference_manifest.json, and registry.json`} />
      <p>
        <code>upload_version</code> validates the version again and sends its three files and{" "}
        <code>registry.json</code> in one commit, so the hub never shows a catalog entry without its files. Then verify
        as an anonymous user:
      </p>
      <CodeBlock code={`alignair models list\nalignair predict --input examples/reads.fasta --out out.tsv --model alignair4-igh-human@<version>`} />
      <p>
        <code>Aligner.push_to_hub(repo_id)</code> remains for a model in a Hugging Face repo of its own: it uploads the
        model file with the <code>model_card.md</code> and <code>reference_manifest.json</code> beside it, in one
        commit. AlignAIR's pretrained models go into the shared catalog instead.
      </p>

      <h2>What gets validated before anything publishes</h2>
      <p>
        Every version is checked by the registry validator: size + SHA-256 match, the artifact is pickle-free (no
        <code> dataconfig</code>/<code>train_state</code> sections) and carries a safe <code>reference_json</code>, the
        card's <code>model_id</code>/<code>model_version</code> match the registry, the reference hashes recompute
        from the embedded reference, and the card and manifest recorded beside it exist with the SHA-256 the entry
        gives. <code>publish_local</code> commits only if it passes, so a broken artifact never
        reaches the upload step.
      </p>
    </>
  ),
};

export const maintainerPages: DocPage[] = [publishing];

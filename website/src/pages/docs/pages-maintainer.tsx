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
        per model.
      </p>
      <ul>
        <li><strong>Anyone can read/download</strong> - <code>alignair models list</code> and <code>predict --model &lt;id&gt;</code> fetch anonymously.</li>
        <li><strong>Only maintainers can publish</strong> - uploading requires a Hugging Face write token. Nothing in the shipped package can write to the repo.</li>
        <li><strong>The catalog auto-updates with no code change</strong> - the CLI reads <code>registry.json</code> live. Re-upload it plus a new artifact and every user sees the change.</li>
      </ul>

      <Callout kind="warning">
        This page is a maintainer runbook. The <code>alignair models</code> CLI exposes only{" "}
        <code>list/get/path/info/update/verify/prune</code> - there is no publish command in the shipped package.
      </Callout>

      <h2>Build the registry locally</h2>
      <CodeBlock code={`python .private/scripts/build_pretrained_registry.py \\\n    --models-dir .private/models --registry-dir .private/registry`} />
      <p>
        This re-cards each source artifact (metadata only - weights and the embedded reference are preserved
        byte-for-byte) and publishes it through the transactional, validator-gated <code>publish_local</code>. Output is{" "}
        <code>.private/registry/</code> with <code>registry.json</code>, a <code>README.md</code>, and the artifacts.
      </p>

      <h2>Upload to Hugging Face (token-gated)</h2>
      <CodeBlock code={`hf upload AlignAIR/AlignAIR-pretrained .private/registry . \\\n    --repo-type model --no-private --exclude "*.lock"`} />
      <p>
        <code>--no-private</code> keeps the repo world-readable and creates it on first upload. Then verify as an
        anonymous user:
      </p>
      <CodeBlock code={`alignair models list\nalignair predict --input examples/reads.fasta --out out.tsv --model alignair-igh-human`} />

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

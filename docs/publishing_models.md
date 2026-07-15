# Publishing pretrained models (maintainer runbook)

AlignAIR distributes pretrained models through a single **public** Hugging Face repository,
[`AlignAIR/AlignAIR-pretrained`](https://huggingface.co/AlignAIR/AlignAIR-pretrained). It holds a
`registry.json` catalog plus one pickle-free `<id>/<version>.alignair` per model.

- **Anyone can read/download** — `alignair models list` and `alignair predict --model <id>` fetch it
  anonymously (no token, no login). The repo is world-readable.
- **Only maintainers can publish** — uploading requires a Hugging Face **write token** with access to
  the `AlignAIR` org. Nothing in the shipped package can write to the repo: the `alignair models` CLI
  exposes only `list/get/path/info/update/verify/prune`, and `Aligner.push_to_hub` refuses without a
  token. Publishing is the deliberate, manual flow below.
- **The catalog auto-updates with no code change** — the CLI reads `registry.json` live on every
  `models list`. Add or retrain a model, re-upload `registry.json` + the new artifact, and every user
  sees it immediately. You never edit or release the `alignair` package to ship a new model.

The default registry is set in code once (`DEFAULT_REGISTRY` in
[`src/alignair/registry/sources.py`](https://github.com/MuteJester/AlignAIR/blob/main/src/alignair/registry/sources.py)); users can override it with
`--registry`, `$ALIGNAIR_REGISTRY`, or `~/.config/alignair/config.toml`.

## One-time setup

1. Ensure the `AlignAIR` org exists on Hugging Face and your account has **write** access to it.
2. Create a **write** token at <https://huggingface.co/settings/tokens> (scope it to the `AlignAIR`
   org if you use fine-grained tokens).
3. Authenticate locally (either works):

   ```bash
   hf auth login            # interactive; stores the token
   # or, per-shell:
   export HF_TOKEN=hf_xxx   # a write token
   ```

## Publish (or re-publish) the models

### 1. Build the registry locally

This re-cards each source artifact with its `model_id`/`version`/`locus`/`species` (metadata only —
weights and the embedded reference are preserved byte-for-byte) and publishes it into a local registry
directory through the transactional, **validator-gated** `publish_local`. It never touches the network.

```bash
python scripts/build_pretrained_registry.py \
    --models-dir .private/models --registry-dir .private/registry
```

Output is `.private/registry/` containing `registry.json`, `README.md` (the repo landing page), and
`<id>/<version>.alignair`. If validation fails for any model, nothing is left half-written — fix the
reported problem and re-run. (The source artifacts and this directory live under `.private/`, which is
gitignored — they are never committed.)

### 2. Upload to Hugging Face (this is the token-gated, maintainer-only step)

```bash
hf upload AlignAIR/AlignAIR-pretrained .private/registry . \
    --repo-type model --no-private --exclude "*.lock"
```

- `.private/registry .` uploads the directory **contents** to the repo root, so `registry.json` lands
  at `…/resolve/main/registry.json` (where the CLI looks).
- `--no-private` keeps the repo world-readable (anonymous downloads). It also creates the repo on first
  upload if it does not exist yet.
- `--exclude "*.lock"` skips the local `registry.json.lock` (a publish-time lock, not a repo file).
- Add `--token hf_xxx` if you did not `hf auth login`.

### 3. Verify as an anonymous user

```bash
# no token / no login — hits the public repo
alignair models list
alignair predict --input examples/reads.fasta --out out.tsv --model alignair-igh-human
```

You should see `alignair-igh-human`, `alignair-igkl-human`, and `alignair-tcrb-human` listed, and the
model download + hash-verify + cache + align run end-to-end.

## Adding or updating a model later (no package release)

Two options — both end with an upload; the CLI catalog updates the moment `registry.json` changes:

- **A new/retrained model in this canonical set:** add or edit an entry in the `MODELS` list at the top
  of [`scripts/build_pretrained_registry.py`](https://github.com/MuteJester/AlignAIR/blob/main/scripts/build_pretrained_registry.py) (bump `version`
  to release an update to an existing id; add a dict for a new id), then re-run step 1 and step 2.
- **A one-off already-carded artifact:** publish it directly into the registry, then upload:

  ```bash
  python scripts/publish_model.py my_model.alignair \
      --id alignair-mouse-igh --version 1.0.0 --registry-dir .private/registry
  hf upload AlignAIR/AlignAIR-pretrained .private/registry . --repo-type model --no-private --exclude "*.lock"
  ```

  The artifact **must be pickle-free** (`alignair train` bundles are; convert a raw `.pt` with
  `alignair convert in.pt out.alignair --dataconfig … --trust-pickle`) and must carry `model_id` /
  `model_version` in its card (set them with `alignair`'s save path, or `model_file.update_card`). The
  validator enforces both, so an invalid artifact is rejected before it can be uploaded.

## What gets validated before anything publishes

Every version is checked by [`registry/validate.py`](https://github.com/MuteJester/AlignAIR/blob/main/src/alignair/registry/validate.py): size + SHA256
match, the artifact is **pickle-free** (no `dataconfig`/`train_state` sections) and carries a safe
`reference_json`, the card's `model_id`/`model_version` match the registry, and the reference hashes
recompute from the embedded reference. `publish_local` runs this against the *staged* state and commits
only if it passes — so a broken artifact never reaches the upload step.

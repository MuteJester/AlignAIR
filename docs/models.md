# Pretrained models

AlignAIR distributes pretrained models from a public hub,
[`AlignAIR/AlignAIR-pretrained`](https://huggingface.co/AlignAIR/AlignAIR-pretrained). You never
download them by hand — pass a model **id** to `--model` and the CLI fetches, verifies, and caches it
on first use. No login or token is required.

## Available models

| id | locus | species | reference |
| --- | --- | --- | --- |
| `alignair-igh-human` | IGH (heavy) | Homo sapiens | GenAIRR OGRDB — V198 / D33 / J7 |
| `alignair-igkl-human` | IGK + IGL (light) | Homo sapiens | GenAIRR — V349 / J18 |
| `alignair-tcrb-human` | TRB (TCR β) | Homo sapiens | GenAIRR — V98 / D3 / J16 |

The live catalog (with install status) is always:

```bash
alignair models list
```

## Using a model

```bash
alignair predict --input reads.fasta --out out.tsv --model alignair-igh-human
```

On first use the model is downloaded, its SHA‑256 is checked against the catalog, and it is cached
locally; subsequent runs load from the cache. Pin an exact version to make a run reproducible:

```bash
alignair predict --input reads.fasta --out out.tsv --model alignair-igh-human@1.0.0
```

`--model` also accepts a **local path** to a `.alignair` file (e.g. one you trained) or an `org/name`
Hugging Face repo id.

## Managing the cache

```bash
alignair models get alignair-tcrb-human      # pre-download (id or id@version)
alignair models info alignair-igh-human      # print the model card
alignair models path alignair-igh-human      # print the cached file path
alignair models verify                       # re-hash installed models against the catalog
alignair models update                       # update installed models to the latest version
alignair models prune                        # remove old cached versions
```

## Offline & private registries

- `--offline` (or `$ALIGNAIR_OFFLINE`) never touches the network — it uses only what is already cached.
- Point at a different registry with `--registry <url>`, the `$ALIGNAIR_REGISTRY` environment variable,
  or `~/.config/alignair/config.toml`. A registry may be a Hugging Face repo (`hf://org/repo`), an
  `https://` mirror, or a local `file://` directory. This is how a lab can host its own private models.

## What a model file contains

Each `.alignair` is **self‑contained and safe to load**: it embeds the trained weights plus the germline
reference (allele sequences + anchors), fingerprinted so the reference cannot drift from the weights. It
loads **without executing any pickle**, so downloading and running a model never runs arbitrary code.
Because the reference is embedded, a model is a fixed‑reference classifier — see the
[model contract](model_contract.md).

Every prediction run also writes a `<out>.run.json` provenance sidecar recording the model id, version,
and fingerprint, so a result can always be traced back to the exact model that produced it.

!!! note "Publishing models"
    Uploading to the hub is a maintainer‑only, token‑gated step — nothing in the installed CLI can
    write to the repository. See [Publishing models](publishing_models.md).

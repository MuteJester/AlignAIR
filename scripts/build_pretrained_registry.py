"""Maintainer tool: build the AlignAIR pretrained-model registry directory from local .alignair
artifacts, ready to upload to the public HuggingFace repo (``AlignAIR/AlignAIR-pretrained``).

For each model it (1) re-cards the artifact with proper ``model_id``/``model_version``/``locus``/
``species`` (metadata only — the weights/reference are preserved byte-for-byte), then (2) publishes it
into the registry directory via the transactional, validator-gated ``publish_local``. The result is a
``<registry-dir>/`` holding ``registry.json`` + ``<id>/<version>.alignair`` — exactly the layout the
CLI reads. UPLOADING that directory to HuggingFace is a separate, token-gated maintainer step (see
docs/publishing_models.md); this script never touches the network.

    python scripts/build_pretrained_registry.py \
        --models-dir .private/models --registry-dir .private/registry

Add or retrain a model later by adding an entry to MODELS (or calling scripts/publish_model.py on a
single already-carded artifact) and re-running — the CLI catalog updates from registry.json with no
package code change once the new registry.json is uploaded.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

from alignair import model_file as mf
from alignair.registry.publish import publish_local

# The published catalog. `source` is the local .alignair (a pickle-free / calibrated inference artifact);
# `card` is the metadata stamped onto it before publishing. Bump `version` (semver) to release an update
# to an existing id; add a new dict to release a new model.
MODELS = [
    {"source": "alignair_igh_v1.cal.alignair", "id": "alignair-igh-human", "version": "1.0.0",
     "card": {"locus": "IGH", "species": "Homo sapiens", "receptor": "IG"},
     "description": "Human IGH (heavy chain) aligner — calibrated. GenAIRR OGRDB reference (V198/D33/J7)."},
    {"source": "alignair_igkl_v1.cal.alignair", "id": "alignair-igkl-human", "version": "1.0.0",
     "card": {"locus": "IGK,IGL", "species": "Homo sapiens", "receptor": "IG"},
     "description": "Human IGK+IGL (light chain) aligner — calibrated. GenAIRR reference (V349/J18)."},
    {"source": "alignair_tcrb_v1.cal.alignair", "id": "alignair-tcrb-human", "version": "1.0.0",
     "card": {"locus": "TRB", "species": "Homo sapiens", "receptor": "TR"},
     "description": "Human TRB (TCR beta) aligner — calibrated. GenAIRR reference (V98/D3/J16)."},
]

_MIN_ALIGNAIR = "2.0.0"          # informational floor recorded in the card

_README = """\
---
license: gpl-3.0
library_name: alignair
tags:
  - immunogenomics
  - AIRR
  - antibody
  - tcr
  - vdj
---

# AlignAIR pretrained models

Official pretrained models for [**AlignAIR**](https://github.com/MuteJester/AlignAIR), a neural aligner
for immunoglobulin (IG) and T-cell-receptor (TCR) repertoires. This repository is the model **registry**
the AlignAIR CLI reads from: `registry.json` lists every model + version, and each `<id>/<version>.alignair`
is a self-contained, **pickle-free** artifact (weights + embedded germline reference + model card).

## Use from the CLI (no manual download)

```bash
pip install "AlignAIR[cli]"
alignair models list                       # this catalog, fetched live
alignair predict --input reads.fasta --out out.tsv --model {default_model}
```

`--model <id>` downloads + hash-verifies + caches the artifact automatically. Pin a version with
`--model <id>@<version>`.

## Models

{model_table}

Each artifact loads without executing any pickle. Reads are anonymous; publishing is maintainer-only.
"""


def _write_repo_readme(registry_dir: str, published: list) -> None:
    rows = ["| id | version | locus | species | description |", "| --- | --- | --- | --- | --- |"]
    for s in published:
        c = s["card"]
        rows.append(f"| `{s['id']}` | {s['version']} | {c['locus']} | {c['species']} | {s['description']} |")
    default = published[0]["id"] if published else "alignair-igh-human"
    with open(os.path.join(registry_dir, "README.md"), "w") as f:
        f.write(_README.format(default_model=default, model_table="\n".join(rows)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", default=".private/models", help="where the source .alignair live")
    ap.add_argument("--registry-dir", default=".private/registry", help="registry directory to build")
    a = ap.parse_args()

    all_problems, published = [], []
    with tempfile.TemporaryDirectory(prefix="alignair-recard-") as staging:
        for spec in MODELS:
            src = os.path.join(a.models_dir, spec["source"])
            if not os.path.exists(src):
                print(f"SKIP {spec['id']}: source not found: {src}")
                all_problems.append(f"{spec['id']}: missing source {src}")
                continue
            carded = os.path.join(staging, f"{spec['id']}-{spec['version']}.alignair")
            updates = {"model_id": spec["id"], "model_version": spec["version"],
                       "created_by_alignair": _MIN_ALIGNAIR, "min_alignair": _MIN_ALIGNAIR,
                       "description": spec["description"], **spec["card"]}
            mf.update_card(src, carded, updates)          # metadata only; weights/reference preserved
            problems = publish_local(carded, spec["id"], spec["version"], a.registry_dir,
                                     description=spec["description"])
            if problems:
                print(f"REJECTED {spec['id']}@{spec['version']}:")
                for p in problems:
                    print(f"  - {p}")
                all_problems += problems
            else:
                size = os.path.getsize(os.path.join(a.registry_dir, spec["id"], f"{spec['version']}.alignair"))
                print(f"published {spec['id']}@{spec['version']}  ({size // 1024 // 1024} MB, validated)")
                published.append(spec)

    if all_problems:
        print(f"\n{len(all_problems)} problem(s) — registry NOT complete.")
        return 1
    _write_repo_readme(a.registry_dir, published)          # HF repo landing page
    print(f"\nregistry built + validated -> {a.registry_dir}")
    print("next: upload it to HuggingFace (see docs/publishing_models.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

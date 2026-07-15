"""Maintainer tool: publish a (pickle-free) .alignair into a local registry directory.

    python scripts/publish_model.py model.alignair --id human-igh --version 2.1.0 --registry-dir ./registry

Copies the artifact, updates registry.json, and runs the validator (aborts on any problem). Uploading
the resulting directory to HuggingFace (`AlignAIR/AlignAIR-pretrained`) is a separate `huggingface-cli
upload` / git-push step (Phase 4). The published artifact MUST be pickle-free — save it with
`include_trusted_pickle=False` (or `alignair convert … --trust-pickle`).
"""
import argparse
import sys

from alignair.registry.publish import publish_local


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", help="the .alignair to publish (must be pickle-free)")
    ap.add_argument("--id", required=True, help="model id, e.g. human-igh")
    ap.add_argument("--version", required=True, help="semver, e.g. 2.1.0")
    ap.add_argument("--registry-dir", required=True, help="local registry directory to write into")
    ap.add_argument("--description", default=None)
    a = ap.parse_args()
    problems = publish_local(a.artifact, a.id, a.version, a.registry_dir, description=a.description)
    if problems:
        print("PUBLISH REJECTED — validation problems:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"published {a.id}@{a.version} -> {a.registry_dir}  (validated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

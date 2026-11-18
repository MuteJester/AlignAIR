"""``freeze-genairr``: copy the importable GenAIRR, named by its content, to run the tests, the audit, model
iteration and claims against instead of the live checkout (``core.freeze``)."""
from __future__ import annotations

import sys

from alignair_assay.core import freeze


def register(sub) -> None:
    """Add the ``freeze-genairr`` subcommand."""
    p = sub.add_parser("freeze-genairr", help="copy the importable GenAIRR into DEST/<digest>/GenAIRR (reused while "
                                              "unchanged) and print the PYTHONPATH that runs it")
    p.add_argument("--dest", default=str(freeze.DEFAULT_DEST),
                   help=f"where frozen copies live (default {freeze.DEFAULT_DEST}, git-ignored)")
    p.set_defaults(func=run)


def run(args) -> int:
    """Freeze GenAIRR (``core.freeze.freeze_genairr``) and print the root, the source and digest, the commit and
    dirty flag of its checkout when it was frozen, and the exact ``PYTHONPATH`` line to use; a checkout with
    uncommitted changes is frozen as it is, with a warning. Returns 0."""
    root = freeze.freeze_genairr(args.dest)
    record = freeze.read_freeze(root) or {}
    commit, dirty = record.get("genairr_git_commit"), record.get("genairr_git_dirty")
    print(f"frozen GenAIRR: {root}")
    print(f"source: {record.get('source')} (version {record.get('genairr_version')}), digest "
          f"{record.get('genairr_source_sha256')}, frozen {record.get('frozen_utc')}")
    if commit is None:
        print("commit: none (not a git checkout), dirty: unknown")
    else:
        print(f"commit: {commit}, dirty: {'yes' if dirty else 'no'}")
    if dirty:
        print("warning: GenAIRR's checkout had uncommitted changes when it was frozen: the copy holds them (its "
              "digest names them), the commit alone does not", file=sys.stderr)
    print(f"PYTHONPATH={root}:src")
    return 0

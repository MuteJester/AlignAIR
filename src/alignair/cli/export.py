"""``alignair export`` - a resumable training checkpoint becomes a pickle-free, distributable bundle.

``alignair export CHECKPOINT OUT_DIR`` wraps ``alignair.train.bundle.export_bundle``: ``OUT_DIR`` gets
``model.alignair`` (no pickle sections), ``model_card.md``, ``reference_manifest.json`` and, unless ``--no-validate``,
``validation_report.json``. ``--model-id`` and ``--model-version`` go into the model file's card, so the catalog's
``publish_local`` accepts the file as it is. The dataconfigs come from the checkpoint (``checkpoint_dataconfigs``).
"""
from __future__ import annotations

import argparse
import sys

from ..errors import ConfigurationError, report_error
from ._errors import fail


def register(sub) -> None:
    p = sub.add_parser("export", help="export a training checkpoint as a pickle-free, distributable bundle")
    p.add_argument("checkpoint", help="a resumable training checkpoint (.alignair with dataconfig/train_state sections)")
    p.add_argument("out_dir", help="the bundle directory to write (refused if it exists, unless --overwrite)")
    p.add_argument("--model-id", default=None, help="the model id recorded in the card (the catalog id)")
    p.add_argument("--model-version", default=None, help="the model version recorded in the card")
    p.add_argument("--description", default="", help="a one-line description recorded in the card")
    p.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True,
                   help="score the model on a fixed held-out stream and write validation_report.json (default: on)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing OUT_DIR")
    p.set_defaults(func=run)


def run(args) -> int:
    from ..train.bundle import checkpoint_dataconfigs, export_bundle
    try:
        dcs, notes = checkpoint_dataconfigs(args.checkpoint)
    except ValueError as e:
        return fail(e, error_type=ConfigurationError, context={"path": args.checkpoint})
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    try:
        path = export_bundle(args.checkpoint, dcs, args.out_dir, model_id=args.model_id,
                             model_version=args.model_version, description=args.description,
                             validate=args.validate, overwrite=args.overwrite)
    except FileExistsError as e:
        return report_error(ConfigurationError(str(e), context={"path": args.out_dir},
                                               hint="Choose a new OUT_DIR or pass --overwrite."))
    print(f"exported pickle-free bundle -> {path}")
    return 0

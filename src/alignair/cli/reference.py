"""``alignair reference`` — work with germline references.

  * ``reference list``   — list the built-in GenAIRR dataconfigs you can train/predict against
                           (optionally filtered by species/chain), so users can discover valid names.
  * ``reference export`` — export a model file's embedded germline FASTA / dataconfig (same as the
                           standalone ``export-reference`` verb, grouped here for discoverability).
"""
from __future__ import annotations


def register(sub) -> None:
    p = sub.add_parser("reference", help="list built-in germline references, or export a model's reference")
    rsub = p.add_subparsers(dest="ref_command", required=True)

    lp = rsub.add_parser("list", help="list the built-in GenAIRR dataconfigs (train/predict references)")
    lp.add_argument("--species", default=None, help="filter by species substring (e.g. Human)")
    lp.add_argument("--chain", default=None, help="filter by chain-type substring (e.g. IGH, TCR)")
    lp.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    lp.set_defaults(func=run_list)

    ep = rsub.add_parser("export", help="export a model file's germline FASTA / dataconfig")
    ep.add_argument("model")
    ep.add_argument("--fasta", help="write the germline V/D/J FASTA here")
    ep.add_argument("--dataconfig", help="write the pickled GenAIRR DataConfig here")
    ep.add_argument("--index", type=int, default=0, help="which embedded dataconfig (multi-chain)")
    ep.set_defaults(func=_run_export)


def _dataconfigs() -> list:
    import GenAIRR.data as gd
    out = []
    for name in sorted(n for n in dir(gd) if not n.startswith("_")):
        obj = getattr(gd, name)
        md = getattr(obj, "metadata", None)
        if md is None:
            continue
        ct = getattr(md, "chain_type", None)
        sp = getattr(md, "species", None)
        out.append({"name": name, "chain_type": str(getattr(ct, "value", ct)),
                    "species": str(getattr(sp, "value", sp)), "has_d": bool(getattr(md, "has_d", False))})
    return out


def run_list(args) -> int:
    rows = _dataconfigs()
    if args.species:
        rows = [r for r in rows if args.species.lower() in r["species"].lower()]
    if args.chain:
        rows = [r for r in rows if args.chain.lower() in (r["chain_type"] + r["name"]).lower()]
    if args.json:
        import json
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{len(rows)} built-in reference(s):")
    print(f"  {'name':28s} {'chain_type':22s} {'species':22s} has_d")
    for r in rows:
        print(f"  {r['name']:28s} {r['chain_type']:22s} {r['species']:22s} {r['has_d']}")
    return 0


def _run_export(args) -> int:
    from .export_reference import run as _export_run
    return _export_run(args)

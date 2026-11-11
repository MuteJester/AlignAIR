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
    lp.add_argument("--chain", default=None,
                    help="filter by a chain type, locus, or a whole token of either or of the name, case-insensitive "
                         "(e.g. IGH, TRB, TCR, BCR_LIGHT_KAPPA, light)")
    lp.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    lp.set_defaults(func=run_list)

    ep = rsub.add_parser("export", help="export a model file's germline FASTA / dataconfig")
    ep.add_argument("model")
    ep.add_argument("--fasta", help="write the germline V/D/J FASTA here")
    ep.add_argument("--dataconfig", help="write the pickled GenAIRR DataConfig here")
    ep.add_argument("--index", type=int, default=0, help="which embedded dataconfig (multi-chain)")
    ep.set_defaults(func=_run_export)


def _dataconfigs() -> list:
    from ..genairr import list_dataconfigs
    return list_dataconfigs()


def run_list(args) -> int:
    rows = _dataconfigs()
    if args.species:
        rows = [r for r in rows if args.species.lower() in r["species"].lower()]
    if args.chain:
        rows = [r for r in rows if _chain_matches(args.chain, r)]
    if args.json:
        import json
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{len(rows)} built-in reference(s):")
    print(f"  {'name':28s} {'chain_type':22s} {'species':22s} has_d")
    for r in rows:
        print(f"  {r['name']:28s} {r['chain_type']:22s} {r['species']:22s} {r['has_d']}")
    return 0


def _chain_matches(query: str, row: dict) -> bool:
    """Whether ``query`` is, case-insensitively, the row's chain type or locus, or one whole ``_``-separated token of
    its chain type or name (``BCR_LIGHT_KAPPA``: BCR, LIGHT, KAPPA; ``HUMAN_TCRB_IMGT``: TCRB), never a substring of
    one (``IGH`` is in ``LIGHT``)."""
    query = query.strip().upper()
    chain, name, locus = (str(row.get(k) or "").upper() for k in ("chain_type", "name", "locus"))
    return query in {chain, locus, *chain.split("_"), *name.split("_")}


def _run_export(args) -> int:
    from .export_reference import run as _export_run
    return _export_run(args)

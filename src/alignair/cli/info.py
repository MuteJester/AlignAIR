"""`alignair info` — print a model's card (metadata only; no weights loaded): a model file, or a registry id or
Hugging Face repo resolved (and cached) first."""
from __future__ import annotations

import json

from ..model_file import embedded_allele_counts, read_metadata
from ._options import MODEL_HELP, add_model_source, resolve_model_path


def register(sub) -> None:
    p = sub.add_parser("info", help="print a model's metadata / model card")
    p.add_argument("model", help=MODEL_HELP)
    p.add_argument("--json", action="store_true", help="print the raw metadata JSON")
    add_model_source(p)
    p.set_defaults(func=run)


def run(args) -> int:
    path = resolve_model_path(args)
    md = read_metadata(path)
    if args.json:
        print(json.dumps(md, indent=2, default=str))
        return 0
    print_card(path, md)
    return 0


def print_card(path, md: dict) -> None:
    """Print the card of the model file at ``path`` (its metadata ``md``)."""
    m, t, ref = md.get("model", {}), md.get("training", {}), md.get("reference", {})
    name = " ".join(str(part) for part in (md.get("model_id"), md.get("model_version")) if part)
    ac = embedded_allele_counts(path)
    print(f"AlignAIR model {name or '(local)'}  (format v{md.get('format_version')}, "
          f"alignair {md.get('created_by_alignair')})")
    print(f"  created:   {md.get('created')}")
    if md.get("description"):
        print(f"  note:      {md.get('description')}")
    if md.get("species") or md.get("locus"):
        print(f"  species/locus: {md.get('species')} / {md.get('locus')}")
    print(f"  alleles:   V={ac.get('v')} D={ac.get('d')} J={ac.get('j')}   params={m.get('param_count')}")
    print(f"  training:  steps={t.get('steps')} total_sequences_seen={t.get('total_sequences_seen')} lr={t.get('lr')}")
    for dc in ref.get("dataconfigs", []):
        print(f"  reference: {dc['name']} ({dc.get('chain_type')}, {dc.get('species')})")
    print(f"  allele order: {(ref.get('allele_order_sha256') or '')[:12]}…")
    print(f"  sections:  {', '.join(md.get('sections', {}))}")

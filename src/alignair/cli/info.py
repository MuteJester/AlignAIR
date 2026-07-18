"""`alignair info` — print an .alignair model file's card (metadata only; no weights loaded)."""
from __future__ import annotations

import json

from ..model_file import read_metadata


def register(sub) -> None:
    p = sub.add_parser("info", help="print an .alignair model file's metadata / model card")
    p.add_argument("model")
    p.add_argument("--json", action="store_true", help="print the raw metadata JSON")
    p.set_defaults(func=run)


def run(args) -> int:
    md = read_metadata(args.model)
    if args.json:
        print(json.dumps(md, indent=2, default=str))
        return 0
    m, t = md.get("model", {}), md.get("training", {})
    ac = m.get("allele_counts", {})
    print(f"AlignAIR model  (format v{md.get('format_version')}, alignair {md.get('alignair_version')})")
    print(f"  created:   {md.get('created')}")
    if md.get("description"):
        print(f"  note:      {md.get('description')}")
    print(f"  alleles:   V={ac.get('v')} D={ac.get('d')} J={ac.get('j')}   params={m.get('param_count')}")
    print(f"  training:  steps={t.get('steps')} total_sequences_seen={t.get('total_sequences_seen')} lr={t.get('lr')}")
    for dc in md.get("reference", {}).get("dataconfigs", []):
        print(f"  reference: {dc['name']} ({dc.get('chain_type')}, {dc.get('species')})")
    print(f"  sections:  {', '.join(md.get('sections', {}))}")
    return 0

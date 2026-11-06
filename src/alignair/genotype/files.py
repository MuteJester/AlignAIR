"""Genotype files: ``{gene: [allele names]}`` as JSON or YAML, read and checked against a reference."""
from __future__ import annotations

from ..errors import InputError


class UnknownAlleleError(InputError):
    """A genotype names alleles the reference does not hold; their sequences belong in the reference."""
    code = "ALIGNAIR_GENOTYPE_UNKNOWN_ALLELE"


def load_genotype(path: str, *, reference=None, drop_unknown: bool = True):
    """Load a genotype from JSON or YAML: ``{gene: [allele names]}``. Returns ``{gene: set(names)}``; with a
    ``reference``, ``(genotype, unknown)``, the names it does not hold dropped (``drop_unknown``) or refused
    (``UnknownAlleleError``)."""
    with open(path) as f:
        text = f.read()
    data = _parse(text, path)
    genotype = {str(g).lower(): {str(n) for n in names} for g, names in data.items()}
    if reference is None:
        return genotype
    unknown: dict[str, set] = {}
    for g, names in list(genotype.items()):
        try:
            ref_names = set(reference.gene(g.upper()).names)
        except (KeyError, AttributeError):                 # gene not in this reference -> skip validation
            continue
        bad = {n for n in names if n not in ref_names}
        if bad:
            unknown[g] = bad
            if drop_unknown:
                genotype[g] = names - bad
            else:
                raise UnknownAlleleError(f"unknown alleles in genotype for {g}: {sorted(bad)} (not in the reference)",
                                         context={"gene": g, "alleles": sorted(bad)},
                                         hint="Give their sequences in --reference.")
    return genotype, unknown


def _parse(text: str, path: str):
    import json
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(text)

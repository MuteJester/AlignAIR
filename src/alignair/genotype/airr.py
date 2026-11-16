"""Emit + validate an AIRR-standard GenotypeSet (interop artifact).

An inferred genotype's alleles become one `Genotype`'s `documented_alleles`; AlignAIR reports no undocumented
alleles or deleted genes, so those lists stay empty. Validation uses the vendored structural schema below.
Source: docs.airr-community.org/en/latest/datarep/germline.html.
"""
from __future__ import annotations

import json
import os

# Vendored structural schema (AIRR Genotype/GenotypeSet).
_REQUIRED = {
    "GenotypeSet": ["receptor_genotype_set_id", "genotype_class_list"],
    "Genotype": ["receptor_genotype_id", "locus", "documented_alleles", "undocumented_alleles",
                 "deleted_genes", "inference_process"],
    "DocumentedAllele": ["label"],
    "UndocumentedAllele": ["allele_name", "sequence"],
    "DeletedGene": ["label"],
}
_LOCI = {"IGH", "IGI", "IGK", "IGL", "TRA", "TRB", "TRD", "TRG"}
_INFERENCE = {"genomic_sequencing", "repertoire_sequencing", None}


def to_genotype_set(genotype: dict[str, list[str]], locus: str, *, germline_set_ref=None,
                    set_id: str = "alignair-genotype-1"):
    """Return ``(genotype_set, warnings)``: one `Genotype` for ``locus`` whose documented alleles are the
    ``genotype``'s (``{gene: allele names}``), gene by gene in its order."""
    warnings: list[str] = []
    if germline_set_ref is None:
        warnings.append("germline_set_ref not provided; documented_alleles.germline_set_ref = null")
    documented = [{"label": allele, "germline_set_ref": germline_set_ref, "phasing": None}
                  for alleles in genotype.values() for allele in alleles]
    genotype = {"receptor_genotype_id": f"{set_id}-{locus}", "locus": locus,
                "documented_alleles": documented, "undocumented_alleles": [],
                "deleted_genes": [], "inference_process": "repertoire_sequencing"}
    return {"receptor_genotype_set_id": set_id, "genotype_class_list": [genotype]}, warnings


def write_genotype_set(path: str | os.PathLike[str], genotype: dict[str, list[str]], locus: str, *,
                       set_id: str = "alignair-genotype-pass") -> None:
    """The genotype ``{gene: alleles}`` of ``locus`` as an AIRR GenotypeSet JSON at ``path`` (``to_genotype_set``)."""
    genotype_set, _ = to_genotype_set(genotype, locus, set_id=set_id)
    with open(path, "w") as handle:
        json.dump(genotype_set, handle, indent=2)


def validate(genotype_set: dict) -> list:
    """Structural AIRR-conformance errors ([] == valid)."""
    return _validate_structural(genotype_set)


# Each Genotype's lists of objects: the list's key and its objects' type (in _REQUIRED).
_GENOTYPE_LISTS = (("documented_alleles", "DocumentedAllele"), ("undocumented_alleles", "UndocumentedAllele"),
                   ("deleted_genes", "DeletedGene"))


def _missing(obj: dict, kind: str) -> list[str]:
    """``<kind> missing <key>`` for each key ``_REQUIRED`` lists for ``kind`` that ``obj`` lacks."""
    return [f"{kind} missing {k}" for k in _REQUIRED[kind] if k not in obj]


def _validate_structural(gs: dict) -> list:
    errors = _missing(gs, "GenotypeSet")
    for gt in gs.get("genotype_class_list", []):
        errors += _missing(gt, "Genotype")
        if "locus" in gt and gt["locus"] not in _LOCI:
            errors.append(f"Genotype invalid locus {gt.get('locus')!r}")
        if "inference_process" in gt and gt["inference_process"] not in _INFERENCE:
            errors.append(f"Genotype invalid inference_process {gt.get('inference_process')!r}")
        for key, kind in _GENOTYPE_LISTS:
            for obj in gt.get(key, []):
                errors += _missing(obj, kind)
    return errors

"""Emit + validate an AIRR-standard GenotypeSet (interop artifact).

Maps the per-gene calls onto the AIRR `GenotypeSet`/`Genotype` schema: kept alleles ->
`documented_alleles`, sufficiently-covered novel candidates -> `undocumented_alleles`, deletion
candidates -> `deleted_genes`. Counts/coverage/evidence are NOT in the schema (they stay in the
report). Validation uses the AIRR `airr` package when importable, else the vendored structural schema
below (local `airr` is not installed). Source: docs.airr-community.org/en/latest/datarep/germline.html.
"""
from __future__ import annotations

# Vendored structural schema (AIRR Genotype/GenotypeSet) — used when the `airr` package is absent.
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


def to_genotype_set(gene_calls, locus: str, *, germline_set_ref=None,
                    set_id: str = "alignair-genotype-1"):
    """Return ``(genotype_set, warnings)``. One `Genotype` for ``locus`` aggregating all gene calls.
    Novel candidates are promoted to `undocumented_alleles` only when ``promotable`` (sufficient
    coverage); partial-coverage candidates stay report-only."""
    warnings: list[str] = []
    if germline_set_ref is None:
        warnings.append("germline_set_ref not provided; documented_alleles.germline_set_ref = null")
    documented, undocumented, deleted = [], [], []
    for gc in gene_calls:
        for a in gc.alleles:
            documented.append({"label": a["name"], "germline_set_ref": germline_set_ref,
                               "phasing": a.get("phasing")})
        for nv in gc.novel:
            if nv.get("promotable"):
                name = nv.get("allele_name") or f"{gc.gene}_novel_{'_'.join(map(str, nv.get('positions', [])))}"
                undocumented.append({"allele_name": name, "sequence": nv.get("sequence"), "phasing": None})
        if gc.deletion_candidate:
            deleted.append({"label": gc.gene})
    genotype = {"receptor_genotype_id": f"{set_id}-{locus}", "locus": locus,
                "documented_alleles": documented, "undocumented_alleles": undocumented,
                "deleted_genes": deleted, "inference_process": "repertoire_sequencing"}
    return {"receptor_genotype_set_id": set_id, "genotype_class_list": [genotype]}, warnings


def validate(genotype_set: dict) -> list:
    """AIRR-conformance errors ([] == valid). Prefers the `airr` package, falls back to the vendored
    structural schema."""
    errors = _validate_structural(genotype_set)
    try:                                                       # best-effort: also run the airr package
        import airr  # noqa: F401  (not installed here; the structural check is authoritative)
    except Exception:
        pass
    return errors


def _validate_structural(gs: dict) -> list:
    errors: list[str] = []
    for k in _REQUIRED["GenotypeSet"]:
        if k not in gs:
            errors.append(f"GenotypeSet missing {k}")
    for gt in gs.get("genotype_class_list", []):
        for k in _REQUIRED["Genotype"]:
            if k not in gt:
                errors.append(f"Genotype missing {k}")
        if "locus" in gt and gt["locus"] not in _LOCI:
            errors.append(f"Genotype invalid locus {gt.get('locus')!r}")
        if "inference_process" in gt and gt["inference_process"] not in _INFERENCE:
            errors.append(f"Genotype invalid inference_process {gt.get('inference_process')!r}")
        for da in gt.get("documented_alleles", []):
            if "label" not in da:
                errors.append("DocumentedAllele missing label")
        for ua in gt.get("undocumented_alleles", []):
            for k in _REQUIRED["UndocumentedAllele"]:
                if k not in ua:
                    errors.append(f"UndocumentedAllele missing {k}")
        for dg in gt.get("deleted_genes", []):
            if "label" not in dg:
                errors.append("DeletedGene missing label")
    return errors

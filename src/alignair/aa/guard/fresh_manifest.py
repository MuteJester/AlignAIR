"""Fresh validation cohort manifest with VHH genus/temporal holdout.

Analogous to Gate-3 MANIFEST but with additional fields for VHH-specific holdout policies.
Provenance only — no raw sequences.
"""
from __future__ import annotations

from dataclasses import dataclass

from alignair.aa.product.applicability.cohort import (
    POSITIVE, GATED_5, GATED_10, STRUCTURAL, DESCRIPTIVE,
    G_IGK_IGL, G_TCR, G_VHH_NONHUMAN, G_IGSF, G_STRUCTURAL,
    G_POS_REAL, G_POS_SIM,
)


@dataclass(frozen=True)
class FreshCategorySpec:
    """Per-category provenance for FRESH cohort with VHH holdout policies. No sequences."""
    key: str
    expected_state: str          # "supported" | "insufficient_evidence" | "unknown_ood"
    group: str
    gate: str                    # POSITIVE | GATED_5 | GATED_10 | DESCRIPTIVE | STRUCTURAL
    source: str
    access: str                  # concrete endpoint / bulk pattern
    locus: str
    species: str
    domain: str
    order: str                   # always "N->C"
    boundary_source: str         # how the complete-VH boundary is set
    license: str
    lineage_key_scheme: str      # how independent lineages are counted
    expected_reason: str | None = None
    note: str = ""
    role_split: str = "both"     # "selection" | "loco_challenge" | "both"
    vhh_genus_holdout: dict | None = None
    vhh_temporal_cutoff_policy: str | None = None


_OAS = "https://opig.stats.ox.ac.uk/webapps/oas/"
_SABDAB = "https://sabdab.opig.stats.ox.ac.uk/api/download/"
_VDJ = "https://vdjserver.org/airr/v1/"
_UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
_THERA = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/static/downloads/TheraSAbDab_SeqStruc_OnlineDownload.csv"

FRESH_MANIFEST: list[FreshCategorySpec] = [
    # ------------------------------------------------------------------ POSITIVES (expected supported)
    FreshCategorySpec("pos_igh_repertoire", "supported", G_POS_REAL, POSITIVE,
                      "OAS unpaired human heavy", f"{_OAS} Species=human&Chain=Heavy (csv.gz units)",
                      "IGH", "human", "VH", "N->C", "AIRR sequence_alignment_aa (database annotation)",
                      "CC-BY 4.0", "subject+study+CDR3/V clonal lineage"),
    FreshCategorySpec("pos_igh_structure", "supported", G_POS_REAL, POSITIVE,
                      "SAbDab human VH structures", f"{_SABDAB}all-summary (human, VH chains)",
                      "IGH", "human", "VH", "N->C", "PDB SEQRES / SAbDab summary (structure)",
                      "CC-BY 4.0", "distinct SABDAB_ID (structure)",
                      note="matched-population positive (decision 2) vs structure-derived negatives"),
    FreshCategorySpec("pos_igh_simulator", "supported", G_POS_SIM, POSITIVE,
                      "GenAIRR HUMAN_IGH_OGRDB (held out from model training)", "in-process simulation",
                      "IGH", "human", "VH", "N->C", "simulator ground-truth boundaries",
                      "n/a", "simulation seed lineage"),
    # ------------------------------------------------------------ NEAR-OOD, combined IGK/IGL (<=5%)
    FreshCategorySpec("igk", "unknown_ood", G_IGK_IGL, GATED_5,
                      "OAS unpaired human light", f"{_OAS} Species=human&Chain=Light (locus=IGK in unit)",
                      "IGK", "human", "V", "N->C", "AIRR locus annotation", "CC-BY 4.0",
                      "subject+study+CDR3/V clonal lineage"),
    FreshCategorySpec("igl", "unknown_ood", G_IGK_IGL, GATED_5,
                      "OAS unpaired human light", f"{_OAS} Species=human&Chain=Light (locus=IGL in unit)",
                      "IGL", "human", "V", "N->C", "AIRR locus annotation", "CC-BY 4.0",
                      "subject+study+CDR3/V clonal lineage"),
    # --------------------------------------------------------------- NEAR-OOD, combined TCR (<=5%)
    FreshCategorySpec("tra", "unknown_ood", G_TCR, GATED_5,
                      "AIRR ADC / VDJServer", f"{_VDJ}repertoire filter locus=TRA (244 repertoires)",
                      "TRA", "human", "V", "N->C", "AIRR locus annotation", "CC-BY 4.0",
                      "repertoire_id ~ subject"),
    FreshCategorySpec("trb", "unknown_ood", G_TCR, GATED_5,
                      "AIRR ADC / VDJServer", f"{_VDJ}repertoire filter locus=TRB (2848 repertoires)",
                      "TRB", "human", "V", "N->C", "AIRR locus annotation", "CC-BY 4.0",
                      "repertoire_id ~ subject"),
    FreshCategorySpec("trg", "unknown_ood", G_TCR, DESCRIPTIVE,
                      "AIRR ADC / STCRDab", f"{_VDJ}repertoire locus=TRG (scarce) + STCRDab",
                      "TRG", "human", "V", "N->C", "AIRR/structure annotation", "CC-BY 4.0",
                      "repertoire_id / structure",
                      note="decision 1: descriptive stratum only, too scarce to formally bound"),
    FreshCategorySpec("trd", "unknown_ood", G_TCR, DESCRIPTIVE,
                      "AIRR ADC / STCRDab", f"{_VDJ}repertoire locus=TRD (scarce) + STCRDab",
                      "TRD", "human", "V", "N->C", "AIRR/structure annotation", "CC-BY 4.0",
                      "repertoire_id / structure",
                      note="decision 1: descriptive stratum only"),
    # ------------------------------------------------------- NEAR-OOD, VHH + non-human VH (<=10%)
    FreshCategorySpec("murine_vh", "unknown_ood", G_VHH_NONHUMAN, GATED_10,
                      "SAbDab murine structures", f"{_SABDAB}all-summary (mus musculus, 1541 IDs)",
                      "IGH", "mouse", "VH", "N->C", "PDB SEQRES (structure)", "CC-BY 4.0",
                      "distinct SABDAB_ID"),
    FreshCategorySpec("vhh", "unknown_ood", G_VHH_NONHUMAN, GATED_10,
                      "SAbDab all-single-domain-summary (camelid)", f"{_SABDAB}all-single-domain-summary (camelid, ~1107 IDs)",
                      "n/a", "camelid", "VHH", "N->C", "PDB SEQRES (structure)", "CC-BY 4.0",
                      "distinct SABDAB_ID",
                      note="single-source (SAbDab); genus holdout; limitation retained",
                      vhh_genus_holdout={"selection": ["lama glama"], "loco_challenge": ["vicugna pacos", "camelus dromedarius", "camelus bactrianus"]},
                      vhh_temporal_cutoff_policy="genus-only holdout (selection=L. glama, challenge=non-L. glama camelids); deposition date is a soft ordering preference within the challenge, NOT a hard cutoff — the pre-split Gate-3 VHH span the latest depositions, leaving no post-cutoff window"),
    FreshCategorySpec("chimeric_murine_variable", "unknown_ood", G_VHH_NONHUMAN, GATED_10,
                      "Thera-SAbDab (Genetics=Chimeric)", _THERA,
                      "IGH", "mouse-variable", "VH", "N->C", "Thera-SAbDab annotation", "research-use",
                      "distinct INN"),
    # ------------------------------------------------------------------- NEAR-OOD, IgSF (<=10%)
    FreshCategorySpec("igsf_vset", "unknown_ood", G_IGSF, GATED_10,
                      "InterPro IPR013106 x UniProt (reviewed, non-IG/TR)",
                      f"{_UNIPROT}?query=(xref:IPR013106) AND reviewed:true NOT (gene:IG*V* OR gene:TR*V*)",
                      "n/a", "cross-species", "V-set", "N->C", "InterPro V-set domain coords (excised)",
                      "CC-BY 4.0", "protein family + sequence-identity cluster",
                      note="decision 3: cross-species orthologs allowed (~735 raw) for powering"),
    # -------------------------------------- POSITIVE CHALLENGE: human-framework / humanized designs
    FreshCategorySpec("design_human_framework", "supported", G_POS_REAL, POSITIVE,
                      "Thera-SAbDab (Genetically human)", _THERA,
                      "IGH", "human", "VH", "N->C", "Thera-SAbDab Genetics annotation", "research-use",
                      "distinct INN", note="positive challenge; Phase-3 16 excluded downstream"),
    FreshCategorySpec("humanized", "supported", G_POS_REAL, POSITIVE,
                      "Thera-SAbDab (Humanised)", _THERA,
                      "IGH", "human-framework", "VH", "N->C", "Thera-SAbDab Genetics annotation",
                      "research-use", "distinct INN", note="positive, reported as a separate challenge slice"),
    # ------------------------------------------ STRUCTURAL negatives (expected insufficient_evidence)
    FreshCategorySpec("cdr3_only", "insufficient_evidence", G_STRUCTURAL, STRUCTURAL,
                      "deterministic transform", "excise CDR3(parent)",
                      "IGH", "human", "CDR3", "N->C", "derived from parent", "inherit parent",
                      "parent lineage", expected_reason="too_short"),
    FreshCategorySpec("v_only", "insufficient_evidence", G_STRUCTURAL, STRUCTURAL,
                      "deterministic transform", "V-region(parent)",
                      "IGH", "human", "V-only", "N->C", "derived from parent", "inherit parent",
                      "parent lineage", expected_reason="missing_j_anchor"),
    FreshCategorySpec("j_only", "insufficient_evidence", G_STRUCTURAL, STRUCTURAL,
                      "deterministic transform", "J-region(parent)",
                      "IGH", "human", "J-only", "N->C", "derived from parent", "inherit parent",
                      "parent lineage", expected_reason="missing_v_anchor"),
    FreshCategorySpec("leader_vh", "insufficient_evidence", G_STRUCTURAL, STRUCTURAL,
                      "deterministic transform", "leader+parent",
                      "IGH", "human", "leader+VH", "N->C", "derived from parent", "inherit parent",
                      "parent lineage", expected_reason="leader_or_constant_contamination"),
    FreshCategorySpec("vh_constant", "insufficient_evidence", G_STRUCTURAL, STRUCTURAL,
                      "deterministic transform", "parent+CH1",
                      "IGH", "human", "VH+constant", "N->C", "derived from parent", "inherit parent",
                      "parent lineage", expected_reason="leader_or_constant_contamination"),
]

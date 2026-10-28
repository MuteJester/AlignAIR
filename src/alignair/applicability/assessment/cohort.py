"""Gate-3 applicability cohort — category provenance manifest + per-sequence entry type.

The MANIFEST is **provenance only** (no raw sequences; hydration writes those to a git-ignored store).
It is grounded in the 2026-07-24 data-source reconnaissance and four settled cohort-design decisions:

  1. **gamma/delta TCR (TRG/TRD): descriptive stratum, NOT formally gated.** Genuine biological scarcity
     (2 repertoires / 22 structures); the combined-TCR <=5% gate is carried by TRA/TRB, with TRG/TRD
     present but reported descriptively.
  2. **Positives span BOTH bulk-repertoire (OAS) AND structure (SAbDab human VH)** so the membership
     detector cannot key on a repertoire-vs-structure source artifact instead of biology.
  3. **IgSF negatives allow cross-species reviewed orthologs** (human-only ~150 is marginal; cross-species
     ~735 gives margin) — still valid non-human-IGH-VH negatives.
  4. **VHH: SAbDab camelid structures only** (CC-BY 4.0); AVIDa-hIL6 dropped (CC BY-NC + single donor).

Ground truth (locus/species/domain/boundaries) is the DECLARED database annotation, never the Stage-1
completeness rules (independent-labels requirement, spec Gate-3 §5/§6).
"""
from __future__ import annotations

from dataclasses import dataclass

# ------------------------------------------------------------------ gate kinds
POSITIVE = "positive"        # expected `supported`; FRR-gated (<=10% upper bound), reported real vs sim
GATED_5 = "gate_5pct"        # near-OOD; FAR 95% upper bound must be <= 5%
GATED_10 = "gate_10pct"      # near-OOD; FAR 95% upper bound must be <= 10%
DESCRIPTIVE = "descriptive"  # present but NOT formally gated (gamma/delta TCR, de-novo)
STRUCTURAL = "structural"    # expected `insufficient_evidence` via the Stage-1 completeness gate

# ------------------------------------------------------------------ gating groups
# The 95% FAR upper-bound gate applies to these COMBINED groups (spec Gate-3 amendment 2):
G_IGK_IGL = "igk_igl"          # combined IGK/IGL, <=5% (both loci must be present)
G_TCR = "tcr"                    # combined TCR, <=5% (carried by TRA/TRB; gamma/delta descriptive)
G_VHH_NONHUMAN = "vhh_nonhuman"  # VHH + non-human VH (murine, murine-variable chimera), <=10%
G_IGSF = "igsf"                  # IgSF V-set (non-IG/TR), <=10%
G_STRUCTURAL = "structural"      # structural negatives -> insufficient_evidence
G_POS_REAL = "positive_real"     # real human VH (repertoire + structure)
G_POS_SIM = "positive_sim"       # simulator human VH (GenAIRR held-out)
G_DENOVO = "denovo"              # de-novo/PLM (descriptive; no clean source)


@dataclass(frozen=True)
class CategorySpec:
    """Per-category provenance + expected verdict. No sequences — provenance only."""
    key: str
    expected_state: str          # "supported" | "insufficient_evidence" | "unknown_ood"
    group: str
    gate: str                    # POSITIVE | GATED_5 | GATED_10 | DESCRIPTIVE | STRUCTURAL
    source: str
    access: str                  # concrete endpoint / bulk pattern (from recon)
    locus: str
    species: str
    domain: str
    order: str                   # always "N->C"
    boundary_source: str         # how the complete-VH boundary is set, independent of Stage-1 rules
    license: str
    lineage_key_scheme: str      # how independent lineages are counted (dedup key)
    expected_reason: str | None = None
    note: str = ""


_OAS = "https://opig.stats.ox.ac.uk/webapps/oas/"
_SABDAB = "https://sabdab.opig.stats.ox.ac.uk/api/download/"
_VDJ = "https://vdjserver.org/airr/v1/"
_UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
_THERA = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/static/downloads/TheraSAbDab_SeqStruc_OnlineDownload.csv"

MANIFEST: list[CategorySpec] = [
    # ------------------------------------------------------------------ POSITIVES (expected supported)
    CategorySpec("pos_igh_repertoire", "supported", G_POS_REAL, POSITIVE,
                 "OAS unpaired human heavy", f"{_OAS} Species=human&Chain=Heavy (csv.gz units)",
                 "IGH", "human", "VH", "N->C", "AIRR sequence_alignment_aa (database annotation)",
                 "CC-BY 4.0", "subject+study+CDR3/V clonal lineage"),
    CategorySpec("pos_igh_structure", "supported", G_POS_REAL, POSITIVE,
                 "SAbDab human VH structures", f"{_SABDAB}all-summary (human, VH chains)",
                 "IGH", "human", "VH", "N->C", "PDB SEQRES / SAbDab summary (structure)",
                 "CC-BY 4.0", "distinct SABDAB_ID (structure)",
                 note="matched-population positive (decision 2) vs structure-derived negatives"),
    CategorySpec("pos_igh_simulator", "supported", G_POS_SIM, POSITIVE,
                 "GenAIRR HUMAN_IGH_OGRDB (held out from model training)", "in-process simulation",
                 "IGH", "human", "VH", "N->C", "simulator ground-truth boundaries",
                 "n/a", "simulation seed lineage"),
    # ------------------------------------------------------------ NEAR-OOD, combined IGK/IGL (<=5%)
    CategorySpec("igk", "unknown_ood", G_IGK_IGL, GATED_5,
                 "OAS unpaired human light", f"{_OAS} Species=human&Chain=Light (locus=IGK in unit)",
                 "IGK", "human", "V", "N->C", "AIRR locus annotation", "CC-BY 4.0",
                 "subject+study+CDR3/V clonal lineage"),
    CategorySpec("igl", "unknown_ood", G_IGK_IGL, GATED_5,
                 "OAS unpaired human light", f"{_OAS} Species=human&Chain=Light (locus=IGL in unit)",
                 "IGL", "human", "V", "N->C", "AIRR locus annotation", "CC-BY 4.0",
                 "subject+study+CDR3/V clonal lineage"),
    # --------------------------------------------------------------- NEAR-OOD, combined TCR (<=5%)
    CategorySpec("tra", "unknown_ood", G_TCR, GATED_5,
                 "AIRR ADC / VDJServer", f"{_VDJ}repertoire filter locus=TRA (244 repertoires)",
                 "TRA", "human", "V", "N->C", "AIRR locus annotation", "CC-BY 4.0",
                 "repertoire_id ~ subject"),
    CategorySpec("trb", "unknown_ood", G_TCR, GATED_5,
                 "AIRR ADC / VDJServer", f"{_VDJ}repertoire filter locus=TRB (2848 repertoires)",
                 "TRB", "human", "V", "N->C", "AIRR locus annotation", "CC-BY 4.0",
                 "repertoire_id ~ subject"),
    CategorySpec("trg", "unknown_ood", G_TCR, DESCRIPTIVE,
                 "AIRR ADC / STCRDab", f"{_VDJ}repertoire locus=TRG (scarce) + STCRDab",
                 "TRG", "human", "V", "N->C", "AIRR/structure annotation", "CC-BY 4.0",
                 "repertoire_id / structure",
                 note="decision 1: descriptive stratum only, too scarce to formally bound"),
    CategorySpec("trd", "unknown_ood", G_TCR, DESCRIPTIVE,
                 "AIRR ADC / STCRDab", f"{_VDJ}repertoire locus=TRD (scarce) + STCRDab",
                 "TRD", "human", "V", "N->C", "AIRR/structure annotation", "CC-BY 4.0",
                 "repertoire_id / structure",
                 note="decision 1: descriptive stratum only"),
    # ------------------------------------------------------- NEAR-OOD, VHH + non-human VH (<=10%)
    CategorySpec("murine_vh", "unknown_ood", G_VHH_NONHUMAN, GATED_10,
                 "SAbDab murine structures", f"{_SABDAB}all-summary (mus musculus, 1541 IDs)",
                 "IGH", "mouse", "VH", "N->C", "PDB SEQRES (structure)", "CC-BY 4.0",
                 "distinct SABDAB_ID"),
    CategorySpec("vhh", "unknown_ood", G_VHH_NONHUMAN, GATED_10,
                 "SAbDab single-domain (camelid)", f"{_SABDAB}all-single-domain-summary (camelid, ~1107 IDs)",
                 "n/a", "camelid", "VHH", "N->C", "PDB SEQRES (structure)", "CC-BY 4.0",
                 "distinct SABDAB_ID",
                 note="decision 4: SAbDab only; AVIDa-hIL6 dropped (CC BY-NC, single donor)"),
    CategorySpec("chimeric_murine_variable", "unknown_ood", G_VHH_NONHUMAN, GATED_10,
                 "Thera-SAbDab (Genetics=Chimeric)", _THERA,
                 "IGH", "mouse-variable", "VH", "N->C", "Thera-SAbDab annotation", "research-use",
                 "distinct INN"),
    # ------------------------------------------------------------------- NEAR-OOD, IgSF (<=10%)
    CategorySpec("igsf_vset", "unknown_ood", G_IGSF, GATED_10,
                 "InterPro IPR013106 x UniProt (reviewed, non-IG/TR)",
                 f"{_UNIPROT}?query=(xref:IPR013106) AND reviewed:true NOT (gene:IG*V* OR gene:TR*V*)",
                 "n/a", "cross-species", "V-set", "N->C", "InterPro V-set domain coords (excised)",
                 "CC-BY 4.0", "protein family + sequence-identity cluster",
                 note="decision 3: cross-species orthologs allowed (~735 raw) for powering"),
    # -------------------------------------- POSITIVE CHALLENGE: human-framework / humanized designs
    CategorySpec("design_human_framework", "supported", G_POS_REAL, POSITIVE,
                 "Thera-SAbDab (Genetically human)", _THERA,
                 "IGH", "human", "VH", "N->C", "Thera-SAbDab Genetics annotation", "research-use",
                 "distinct INN", note="positive challenge; Phase-3 16 excluded downstream"),
    CategorySpec("humanized", "supported", G_POS_REAL, POSITIVE,
                 "Thera-SAbDab (Humanised)", _THERA,
                 "IGH", "human-framework", "VH", "N->C", "Thera-SAbDab Genetics annotation",
                 "research-use", "distinct INN", note="positive, reported as a separate challenge slice"),
    # ------------------------------------------ STRUCTURAL negatives (expected insufficient_evidence)
    CategorySpec("reversed_vh", "insufficient_evidence", G_STRUCTURAL, STRUCTURAL,
                 "deterministic transform of a complete-VH parent", "reverse(parent)",
                 "IGH", "human", "VH-reversed", "C->N", "derived from parent", "inherit parent",
                 "parent lineage", expected_reason="reversed_order",
                 note="effective N = number of parent lineages, not transforms"),
    CategorySpec("cdr3_only", "insufficient_evidence", G_STRUCTURAL, STRUCTURAL,
                 "deterministic transform", "excise CDR3(parent)",
                 "IGH", "human", "CDR3", "N->C", "derived from parent", "inherit parent",
                 "parent lineage", expected_reason="too_short"),
    CategorySpec("v_only", "insufficient_evidence", G_STRUCTURAL, STRUCTURAL,
                 "deterministic transform", "V-region(parent)",
                 "IGH", "human", "V-only", "N->C", "derived from parent", "inherit parent",
                 "parent lineage", expected_reason="missing_j_anchor"),
    CategorySpec("j_only", "insufficient_evidence", G_STRUCTURAL, STRUCTURAL,
                 "deterministic transform", "J-region(parent)",
                 "IGH", "human", "J-only", "N->C", "derived from parent", "inherit parent",
                 "parent lineage", expected_reason="missing_v_anchor"),
    CategorySpec("leader_vh", "insufficient_evidence", G_STRUCTURAL, STRUCTURAL,
                 "deterministic transform", "leader+parent",
                 "IGH", "human", "leader+VH", "N->C", "derived from parent", "inherit parent",
                 "parent lineage", expected_reason="leader_or_constant_contamination"),
    CategorySpec("vh_constant", "insufficient_evidence", G_STRUCTURAL, STRUCTURAL,
                 "deterministic transform", "parent+CH1",
                 "IGH", "human", "VH+constant", "N->C", "derived from parent", "inherit parent",
                 "parent lineage", expected_reason="leader_or_constant_contamination"),
    # ----------------------------------------------------- DE-NOVO / PLM (descriptive; not a blocker)
    CategorySpec("de_novo", "unknown_ood", G_DENOVO, DESCRIPTIVE,
                 "no clean provenance source found", "per-paper manual harvest (deferred)",
                 "n/a", "synthetic", "antibody-like", "N->C", "paper-declared", "varies",
                 "distinct design", note="deprioritized per recon; not a Phase-0 blocker"),
]


def load_manifest() -> list[CategorySpec]:
    return list(MANIFEST)


@dataclass(frozen=True)
class CohortEntry:
    """One hydrated, labelled sequence. The sequence itself lives in the git-ignored hydration store,
    referenced by ``id``; this entry carries only the label + independent-lineage key used for
    deduplication and feasibility counting."""
    id: str
    category: str
    group: str
    expected_state: str
    lineage_key: str

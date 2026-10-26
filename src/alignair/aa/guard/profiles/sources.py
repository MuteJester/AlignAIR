"""Redistribution-gated profile-source registry for the AA applicability guard's competitive profile-HMM
bank (Stage 2, Task 2.1).

Every required V/J or full-domain profile must resolve to a PUBLISHABLE production path — a
redistribution-compatible source we may ship (with any required attribution), or an exact user-side
hydration recipe that deterministically reproduces the pinned bank — BEFORE any bank is built or any
candidate is trained. If any required profile is unresolved, ``resolve_production_bank`` raises
``NoPublishableProfileSource`` → Stage-2 ``STOP_NO_PUBLISHABLE_PROFILE_SOURCE``.

License evidence is REVIEWED from primary sources (verified 2026-07-27), never inferred from popularity:

- OGRDB / AIRR-C germline sets — **CC0 1.0** (each Zenodo set's own ``licence.txt`` = CC0; e.g. Human
  IGKappa VJ rev_4, DOI 10.5281/zenodo.13923289). Released today: human IG (IGH/IGK/IGL). CC0 → derived
  HMM profiles freely redistributable, no attribution.
- Pfam PF07686 (Immunoglobulin V-set; InterPro IPR013106) — **CC0** (EBI FTP). Covers ``igsf_vset``.
- IMGT/GENE-DB — **CLEARED to CC BY-4.0 by owner (2026-07-27)**. IMGT's terms-of-use page states verbatim:
  *"The IMGT® data are provided to both public and private users under the CC BY-4.0 license."*
  (https://www.imgt.org/about/termsofuse.php, fetched 2026-07-27). CC BY-4.0 permits redistribution AND
  derivative works (our HMM profiles) with **attribution**; a trademark carve-out forbids using the IMGT
  name for promotion without approval. This supersedes IMGT's historically-cited CC BY-NC-ND 4.0 (see the
  2022 IMGT paper) — a relicensing the owner explicitly accepted. IMGT is the released authoritative source
  for human TR (TRA/TRB/TRG/TRD), mouse, and camelid VHH germline, so clearing it resolves those loci.

All required profiles now resolve (CC0 or CC BY-4.0) → Stage-2 GO. Attribution obligations are carried per
source so the bundle can emit correct NOTICE/attribution at ship time.
"""
from __future__ import annotations

from alignair.errors import ModelError

from dataclasses import dataclass, field

# Receptor groups whose profile bank needs a V and a J region; plus the igsf_vset full-domain profile.
# Profile-bank group keys are germline IDENTITIES, not input verdicts: camelid IGH germline is
# 'camelid_igh' (it provides non-human/camelid evidence), NOT proof the input is a VHH. (A fresh-cohort
# record verified as a VHH keeps category 'vhh'; that is a separate namespace from these profile keys.)
RECEPTOR_GROUPS = {"human_igh", "human_igk", "human_igl", "tra", "trb", "trg", "trd", "camelid_igh", "murine_vh"}
REQUIRED_PROFILES = RECEPTOR_GROUPS | {"igsf_vset"}

PENDING = "bound_at_bank_build"      # exact per-file sha256 is pinned when the bank is built (Task 2.2)

# Owner license clearance record (auditable): what was cleared, on what evidence, when.
IMGT_CLEARANCE = {
    "source": "IMGT/GENE-DB",
    "license": "CC-BY-4.0",
    "cleared_by": "owner",
    "cleared_date": "2026-07-27",
    "relicensing_date": "2026-07-01",          # IMGT homepage dates the CC BY-NC-ND -> CC BY-4.0 change here
    "evidence_url": "https://www.imgt.org/about/termsofuse.php",
    "evidence_fetched": "2026-07-27",
    "evidence_quote": "The IMGT® data are provided to both public and private users under the CC BY-4.0 license.",
    "evidence_archive": "Notes and Plans/docs/superpowers/artifacts/2026-07-26-aa-applicability/license_evidence.json",
    "supersedes": "historically-cited CC BY-NC-ND 4.0 (e.g. 2022 IMGT NAR paper)",
    "usage_distinction": "AlignAIR downloads germline DATA and builds profile HMMs locally with pyhmmer "
                         "(vendored HMMER, not an IMGT tool); it does NOT redistribute IMGT-hosted files nor "
                         "invoke IMGT web tools/services (IMGT tool-use terms therefore do not apply).",
    "obligations": {
        "attribution": "Data derived from IMGT/GENE-DB (imgt.org), CC BY-4.0.",
        "modification_notice": "Modified: profile HMMs were built from the germline sequences.",
        "placement": ["shipped bundle NOTICE", "model card", "profile manifest"],
        "no_endorsement": "Attribution must not imply IMGT endorsement; the IMGT name must not be used for "
                          "publicity/business promotion without written approval (trademark).",
    },
}


class NoPublishableProfileSource(ModelError):
    """Raised when one or more REQUIRED profiles have neither a redistributable bundle nor an exact
    user-hydration recipe. Carries the unresolved group names."""
    code = "ALIGNAIR_PROFILE_SOURCE_MISSING"

    def __init__(self, unresolved):
        self.unresolved = sorted(unresolved)
        super().__init__(f"STOP_NO_PUBLISHABLE_PROFILE_SOURCE: unresolved required profiles {self.unresolved}")


@dataclass(frozen=True)
class ProfileSource:
    name: str
    url: str
    version: str                              # exact pinned release / DOI / accession
    regions: frozenset                        # {"v","j"} for receptor groups, {"domain"} for full-domain
    license: str
    redistributable: bool                     # may we redistribute profiles DERIVED from this source?
    role: str                                 # "production" | "comparator_only"
    attribution: str = ""                     # required attribution string, "" if none (CC0)
    input_hashes: dict = field(default_factory=dict)   # region/file -> hash (or PENDING)
    hydration: dict | None = None             # exact user-hydration recipe when not directly bundled
    notes: str = ""


_OGRDB_URL = "https://ogrdb.airr-community.org/germline_sets/Homo%20sapiens"
_PFAM_URL = "https://www.ebi.ac.uk/interpro/entry/pfam/PF07686"
_IMGT_URL = "https://www.imgt.org/genedb/"
_IMGT_ATTR = "Data derived from IMGT/GENE-DB (imgt.org), CC BY-4.0; changes: profile HMMs built from germline."


def _ogrdb_released(name, url, version, input_hashes=None):
    """A CONFIRMED-released OGRDB CC0 set (human IG today) -> redistributable production source, no attribution."""
    return ProfileSource(
        name=name, url=url, version=version, regions=frozenset({"v", "j"}),
        license="CC0-1.0", redistributable=True, role="production", attribution="",
        input_hashes=input_hashes or {"v": PENDING, "j": PENDING},
        notes="OGRDB/AIRR-C released germline set; per-set Zenodo licence.txt = CC0 (verified).")


def _imgt_prod(name, version, why):
    """IMGT germline cleared to CC BY-4.0 -> redistributable production source WITH attribution."""
    return ProfileSource(
        name=name, url=_IMGT_URL, version=version, regions=frozenset({"v", "j"}),
        license="CC-BY-4.0", redistributable=True, role="production", attribution=_IMGT_ATTR,
        input_hashes={"v": PENDING, "j": PENDING}, notes=why)


PROFILE_SOURCES: dict[str, ProfileSource] = {
    # --- CC0 released (no attribution) ----------------------------------------------------------------
    "human_igh": _ogrdb_released("ogrdb_human_ighv_ighj", _OGRDB_URL, "AIRR-C Human IGH set (OGRDB/Zenodo, pin rev)"),
    "human_igk": _ogrdb_released("ogrdb_human_igkv_igkj", _OGRDB_URL,
                                 "AIRR-C Human IGKappa VJ rev_4 (DOI 10.5281/zenodo.13923289)",
                                 input_hashes={"vj_ungapped_fasta_md5": "3a2857d50403f95f8bce3b713fbcc5be",
                                               "v": PENDING, "j": PENDING}),
    "human_igl": _ogrdb_released("ogrdb_human_iglv_iglj", _OGRDB_URL, "AIRR-C Human IGLambda set (OGRDB/Zenodo, pin rev)"),
    "igsf_vset": ProfileSource(
        name="pfam_v_set_PF07686", url=_PFAM_URL, version="Pfam PF07686 (InterPro IPR013106)",
        regions=frozenset({"domain"}), license="CC0", redistributable=True, role="production",
        attribution="", input_hashes={"domain": PENDING}, notes="Pfam-A HMM; Pfam flatfiles/HMMs are CC0 (EBI FTP)."),
    # --- IMGT CC BY-4.0 (cleared 2026-07-27; attribution required) ------------------------------------
    "tra": _imgt_prod("imgt_human_trav_traj", "IMGT/GENE-DB Human TRAV,TRAJ (pin release)",
                      "IMGT cleared CC BY-4.0; OGRDB human TR sets not yet released."),
    "trb": _imgt_prod("imgt_human_trbv_trbj", "IMGT/GENE-DB Human TRBV,TRBJ (pin release)",
                      "IMGT cleared CC BY-4.0; OGRDB human TR sets not yet released."),
    "trg": _imgt_prod("imgt_human_trgv_trgj", "IMGT/GENE-DB Human TRGV,TRGJ (pin release)",
                      "IMGT cleared CC BY-4.0; OGRDB human TR sets not yet released."),
    "trd": _imgt_prod("imgt_human_trdv_trdj", "IMGT/GENE-DB Human TRDV,TRDJ (pin release)",
                      "IMGT cleared CC BY-4.0; OGRDB human TR sets not yet released."),
    "murine_vh": _imgt_prod("imgt_mouse_ighv_ighj", "IMGT/GENE-DB Mus musculus IGHV,IGHJ (pin release)",
                            "IMGT cleared CC BY-4.0; OGRDB mouse release unconfirmed."),
    "camelid_igh": _imgt_prod("imgt_camelid_ighv_ighj", "IMGT/GENE-DB Vicugna pacos IGHV,IGHJ (pin release)",
                              "IMGT cleared CC BY-4.0; camelid IGH germline (non-human/camelid evidence, "
                              "not a biologically definitive 'VHH' profile)."),
}


@dataclass(frozen=True)
class ProductionBankResolution:
    mode: str                                 # "redistributable_bundle" | "exact_user_hydration"
    all_required_profiles_resolved: bool
    resolutions: dict                         # group -> (path_mode, ProfileSource)
    attributions: tuple = ()                  # distinct attribution strings the bundle must emit


@dataclass(frozen=True)
class HydrationResolution:
    """An exact user-side hydration binding: pinned inputs, hydration script, and reproduced bank output."""
    input_sha256: str
    bank_sha256: str
    hydration_script_sha256: str
    acquisition_url: str
    first_run_network: str
    offline_after_hydration: bool
    failure_semantics: str


def resolve_production_bank() -> ProductionBankResolution:
    """Resolve every REQUIRED profile to a publishable path, or raise NoPublishableProfileSource.

    A profile resolves as ``redistributable_bundle`` if its source is production + redistributable, or as
    ``exact_user_hydration`` if it carries a pinned hydration recipe. A ``comparator_only`` source with no
    hydration recipe is UNRESOLVED."""
    unresolved, resolutions, attributions = [], {}, set()
    for group in REQUIRED_PROFILES:
        src = PROFILE_SOURCES[group]
        if src.role == "production" and src.redistributable:
            resolutions[group] = ("redistributable_bundle", src)
            if src.attribution:
                attributions.add(src.attribution)
        elif src.hydration is not None:
            resolutions[group] = ("exact_user_hydration", src)
        else:
            unresolved.append(group)
    if unresolved:
        raise NoPublishableProfileSource(unresolved)
    modes = {m for m, _ in resolutions.values()}
    mode = "redistributable_bundle" if modes == {"redistributable_bundle"} else "exact_user_hydration"
    return ProductionBankResolution(mode=mode, all_required_profiles_resolved=True,
                                    resolutions=resolutions, attributions=tuple(sorted(attributions)))


def synthetic_hydration_resolution() -> HydrationResolution:
    """A SYNTHETIC exact-hydration binding used to test the reproducibility contract (pins inputs, the
    hydration script, and the exact reproduced bank output). Not a real production bank."""
    return HydrationResolution(
        input_sha256="0" * 64, bank_sha256="1" * 64, hydration_script_sha256="2" * 64,
        acquisition_url="https://ogrdb.airr-community.org/germline_sets/<pinned-doi>",
        first_run_network="download pinned release once, verify input_sha256",
        offline_after_hydration=True,
        failure_semantics="hard-fail on input_sha256 or bank_sha256 mismatch; never substitute a bank")


def source_matrix() -> dict:
    """Counts/provenance-only matrix for review (no sequences). Includes the resolution attempt outcome."""
    def _row(g, s):
        return {"group": g, "name": s.name, "url": s.url, "version": s.version,
                "regions": sorted(s.regions), "license": s.license, "redistributable": s.redistributable,
                "role": s.role, "attribution": s.attribution, "input_hashes": s.input_hashes, "notes": s.notes}
    matrix = {"required_profiles": sorted(REQUIRED_PROFILES),
              "sources": {g: _row(g, PROFILE_SOURCES[g]) for g in sorted(PROFILE_SOURCES)},
              "imgt_clearance": IMGT_CLEARANCE}
    try:
        res = resolve_production_bank()
        matrix["resolution"] = {"decision": "GO", "mode": res.mode,
                                "all_required_profiles_resolved": True, "attributions": list(res.attributions)}
    except NoPublishableProfileSource as e:
        matrix["resolution"] = {"decision": "STOP_NO_PUBLISHABLE_PROFILE_SOURCE", "unresolved": e.unresolved}
    return matrix

"""Redistribution-gated profile-source registry for the AA applicability guard's competitive profile-HMM
bank (Stage 2, Task 2.1).

Every required V/J or full-domain profile must resolve to a PUBLISHABLE production path — either a
redistribution-compatible bundle we may ship, or an exact user-side hydration recipe that deterministically
reproduces the pinned bank — BEFORE any bank is built or any candidate is trained. IMGT is registered
``comparator_only`` until redistribution is explicitly cleared. If any required profile is unresolved,
``resolve_production_bank`` raises ``NoPublishableProfileSource`` → Stage-2 ``STOP_NO_PUBLISHABLE_PROFILE_SOURCE``.

License evidence is REVIEWED from primary sources (verified 2026-07-27), never inferred from popularity:

- OGRDB / AIRR-C germline sets — **CC0 1.0** (each Zenodo set carries its own ``licence.txt`` = CC0; e.g.
  Human IGKappa VJ rev_4, DOI 10.5281/zenodo.13923289). CC0 permits redistribution of the data AND of
  derived HMM profiles with no attribution/share-alike. **Released today: human IG only (IGH, IGK, IGL).**
  Human **TR** reference sets are explicitly "still in development" on OGRDB (not released/pinnable); **mouse**
  released status is unconfirmed from public evidence. So OGRDB currently resolves ONLY human IG.
- Pfam PF07686 (Immunoglobulin V-set; InterPro IPR013106) — **CC0** (Pfam flatfiles/HMMs, EBI FTP). Covers
  the ``igsf_vset`` full-domain profile.
- IMGT/GENE-DB — authoritative for all IG/TR incl. camelid AND mouse, but redistribution-restricted →
  ``comparator_only`` until explicitly cleared. It is the only released authoritative source for TR, mouse,
  and camelid today.
- Camelid VHH germline — **no CC0/permissive curated set exists** (absent from OGRDB, whose species are
  human/macaque/mouse/trout; only IMGT + scattered public-domain GenBank submissions).

RESOLVED (CC0 production) today: human_igh, human_igk, human_igl, igsf_vset. UNRESOLVED (only IMGT
comparator, or OGRDB-in-development, or no source): tra, trb, trg, trd, murine_vh, vhh → Stage-2 STOP.

NOTE (plan deviation): the plan's Task-2.1 test optimistically assumed resolution succeeds. The owner's
governing directive — "any unresolved required profile produces STOP_NO_PUBLISHABLE_PROFILE_SOURCE" —
takes precedence, so ``resolve_production_bank`` raises here. See ``test_profile_sources.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Receptor groups whose profile bank needs a V and a J region; plus the igsf_vset full-domain profile.
RECEPTOR_GROUPS = {"human_igh", "human_igk", "human_igl", "tra", "trb", "trg", "trd", "vhh", "murine_vh"}
REQUIRED_PROFILES = RECEPTOR_GROUPS | {"igsf_vset"}

PENDING = "bound_at_bank_build"      # exact per-file sha256 is pinned when the bank is built (Task 2.2)


class NoPublishableProfileSource(Exception):
    """Raised when one or more REQUIRED profiles have neither a redistributable bundle nor an exact
    user-hydration recipe (IMGT/comparator_only does not count). Carries the unresolved group names."""

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
    input_hashes: dict = field(default_factory=dict)   # region/file -> hash (or PENDING)
    hydration: dict | None = None             # exact user-hydration recipe when not directly bundled
    notes: str = ""


# --- Registry: reviewed evidence per required profile ---------------------------------------------------
_OGRDB_URL = "https://ogrdb.airr-community.org/germline_sets/Homo%20sapiens"
_PFAM_URL = "https://www.ebi.ac.uk/interpro/entry/pfam/PF07686"


def _ogrdb_released(name, url, version, input_hashes=None):
    """A CONFIRMED-released OGRDB CC0 set (human IG today) -> redistributable production source."""
    return ProfileSource(
        name=name, url=url, version=version, regions=frozenset({"v", "j"}),
        license="CC0-1.0", redistributable=True, role="production",
        input_hashes=input_hashes or {"v": PENDING, "j": PENDING},
        notes="OGRDB/AIRR-C released germline set; per-set Zenodo licence.txt = CC0 (verified).")


def _imgt_only(name, version, why):
    """A required profile whose only released authoritative source is IMGT (comparator_only) -> UNRESOLVED."""
    return ProfileSource(
        name=name, url="https://www.imgt.org/genedb/", version=version,
        regions=frozenset({"v", "j"}), license="IMGT terms (restricted)", redistributable=False,
        role="comparator_only", notes=why)


PROFILE_SOURCES: dict[str, ProfileSource] = {
    # --- resolved: CC0 released, redistributable production -------------------------------------------
    "human_igh": _ogrdb_released("ogrdb_human_ighv_ighj", _OGRDB_URL, "AIRR-C Human IGH set (OGRDB/Zenodo, pin rev)"),
    "human_igk": _ogrdb_released("ogrdb_human_igkv_igkj", _OGRDB_URL,
                                 "AIRR-C Human IGKappa VJ rev_4 (DOI 10.5281/zenodo.13923289)",
                                 input_hashes={"vj_ungapped_fasta_md5": "3a2857d50403f95f8bce3b713fbcc5be",
                                               "v": PENDING, "j": PENDING}),
    "human_igl": _ogrdb_released("ogrdb_human_iglv_iglj", _OGRDB_URL, "AIRR-C Human IGLambda set (OGRDB/Zenodo, pin rev)"),
    "igsf_vset": ProfileSource(
        name="pfam_v_set_PF07686", url=_PFAM_URL, version="Pfam PF07686 (InterPro IPR013106)",
        regions=frozenset({"domain"}), license="CC0", redistributable=True, role="production",
        input_hashes={"domain": PENDING}, notes="Pfam-A HMM; Pfam flatfiles/HMMs are CC0 (EBI FTP)."),
    # --- unresolved: OGRDB TR sets are "still in development" (not released) -> only IMGT (comparator) --
    "tra": _imgt_only("imgt_human_trav_traj", "IMGT/GENE-DB Human TRAV,TRAJ",
                      "OGRDB human TR sets are still in development (not released/pinnable); only IMGT "
                      "(comparator_only) exists today. UNRESOLVED. CC0 resolves once OGRDB releases TRA."),
    "trb": _imgt_only("imgt_human_trbv_trbj", "IMGT/GENE-DB Human TRBV,TRBJ",
                      "OGRDB human TR sets are still in development; only IMGT (comparator_only). UNRESOLVED."),
    "trg": _imgt_only("imgt_human_trgv_trgj", "IMGT/GENE-DB Human TRGV,TRGJ",
                      "OGRDB human TR sets are still in development; only IMGT (comparator_only). UNRESOLVED."),
    "trd": _imgt_only("imgt_human_trdv_trdj", "IMGT/GENE-DB Human TRDV,TRDJ",
                      "OGRDB human TR sets are still in development; only IMGT (comparator_only). UNRESOLVED."),
    # --- unresolved: mouse release status unconfirmed on OGRDB -> only IMGT (comparator) ---------------
    "murine_vh": _imgt_only("imgt_mouse_ighv_ighj", "IMGT/GENE-DB Mus musculus IGHV,IGHJ",
                            "Mouse released-set status unconfirmed on OGRDB (species page shows no affirmed "
                            "sequences); only IMGT (comparator_only) confirmed today. UNRESOLVED pending an "
                            "OGRDB mouse release."),
    # --- unresolved: no curated camelid germline set exists at all -------------------------------------
    "vhh": _imgt_only("imgt_camelid_ighv_ighj", "IMGT/GENE-DB Lama/Vicugna/Camelus IGHV,IGHJ",
                      "No redistributable curated camelid germline set exists (absent from OGRDB). IMGT is "
                      "comparator_only; scattered public-domain GenBank submissions are not an authoritative "
                      "curated pinned set. UNRESOLVED."),
}

# IMGT registered explicitly as a comparator (human IG/TR), never a production source until cleared.
PROFILE_SOURCES_COMPARATORS: dict[str, ProfileSource] = {
    "imgt_gene_db": ProfileSource(
        name="imgt_gene_db", url="https://www.imgt.org/genedb/", version="IMGT/GENE-DB (research comparator)",
        regions=frozenset({"v", "j"}), license="IMGT terms (restricted)", redistributable=False,
        role="comparator_only", notes="Local research comparator only until redistribution is explicitly cleared."),
}


@dataclass(frozen=True)
class ProductionBankResolution:
    mode: str                                 # "redistributable_bundle" | "exact_user_hydration"
    all_required_profiles_resolved: bool
    resolutions: dict                         # group -> (path_mode, ProfileSource)


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
    ``exact_user_hydration`` if it carries a pinned hydration recipe. A ``comparator_only`` source (e.g. IMGT)
    with no hydration recipe is UNRESOLVED."""
    unresolved, resolutions = [], {}
    for group in REQUIRED_PROFILES:
        src = PROFILE_SOURCES[group]
        if src.role == "production" and src.redistributable:
            resolutions[group] = ("redistributable_bundle", src)
        elif src.hydration is not None:
            resolutions[group] = ("exact_user_hydration", src)
        else:
            unresolved.append(group)
    if unresolved:
        raise NoPublishableProfileSource(unresolved)
    modes = {m for m, _ in resolutions.values()}
    mode = "redistributable_bundle" if modes == {"redistributable_bundle"} else "exact_user_hydration"
    return ProductionBankResolution(mode=mode, all_required_profiles_resolved=True, resolutions=resolutions)


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
                "role": s.role, "input_hashes": s.input_hashes, "notes": s.notes}
    matrix = {"required_profiles": sorted(REQUIRED_PROFILES),
              "sources": {g: _row(g, PROFILE_SOURCES[g]) for g in sorted(PROFILE_SOURCES)},
              "comparators": {k: _row(k, v) for k, v in PROFILE_SOURCES_COMPARATORS.items()}}
    try:
        res = resolve_production_bank()
        matrix["resolution"] = {"decision": "GO", "mode": res.mode, "all_required_profiles_resolved": True}
    except NoPublishableProfileSource as e:
        matrix["resolution"] = {"decision": "STOP_NO_PUBLISHABLE_PROFILE_SOURCE", "unresolved": e.unresolved}
    return matrix

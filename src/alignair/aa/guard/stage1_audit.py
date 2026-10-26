"""Final global Stage-1 audit: combine the three allocation modes, dedup globally after assignment, and
verify every owner-required completion gate before a Stage-1 GO.

Allocation routing:
  * study-based groups (igk_igl, tcr, positive_real) -> study/donor split barrier (study_allocate), all
    together so heavy+light of one study share a role;
  * vhh_nonhuman -> genus holdout (allocate);
  * other groups (igsf, positive_sim) -> deterministic lineage sha-rank split;
  * structural inherits its real-IGH parent's role+lineage (applied upstream, before this audit).

Audit gates: post-dedup counts; counts by role/locus/study/donor; ZERO study/donor/lineage overlap
between roles; IGK/IGL 37/37 balance per subset + powered real-IGH; frozen allocator policy hash
reproduced exactly.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from alignair.aa.guard.allocate import GROUP_GATE_KIND, GUARDBAND_FLOOR, _allocate_vhh_nonhuman, _halve
from alignair.aa.guard.study_allocate import (DIVERSITY_TARGET, GROUP_LOCI, assign_studies,
                                              canonical_study)

STUDY_GROUPS = {"igk_igl", "tcr", "positive_real"}
VHH_GROUP = "vhh_nonhuman"
REQUIRED_GROUPS = {"igk_igl", "tcr", "vhh_nonhuman", "igsf", "structural",
                   "positive_real", "positive_sim"}

# Per-role per-locus donor/study diversity floors for the discriminative 5%-gate loci: a validation slice
# must never rest on a single donor/study. Sourced from the allocator's policy so the objective the greedy
# steers toward and the gate the audit enforces are the SAME contract. Owner directive scoped IGL; IGK
# shares the identical single-donor risk and is held to the same bar (see DIVERSITY_TARGET).
LOCUS_DIVERSITY = DIVERSITY_TARGET


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def global_dedup(records):
    """Global sequence dedup (by sha256(aa)) across all sources; returns (kept, dropped_count)."""
    seen, kept, dropped = set(), [], 0
    for r in records:
        h = _sha(r["aa"])
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        kept.append(r)
    return kept, dropped


def combine_allocation(records, *, vhh_cutoff=None):
    """Route each group to its allocation mode and merge into (selection, loco, study_alloc)."""
    by_group: dict = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)

    selection: dict = {}
    loco: dict = {}

    study_records = [r for r in records if r["group"] in STUDY_GROUPS]
    study_alloc = assign_studies(study_records) if study_records else None
    if study_alloc:
        for g in STUDY_GROUPS:
            if g in study_alloc.selection:
                selection[g] = set(study_alloc.selection[g])
            if g in study_alloc.loco_challenge:
                loco[g] = set(study_alloc.loco_challenge[g])

    if VHH_GROUP in by_group:
        sel, lc, _mode = _allocate_vhh_nonhuman(by_group[VHH_GROUP], vhh_cutoff)
        selection[VHH_GROUP] = sel
        loco[VHH_GROUP] = lc

    for g, recs in by_group.items():
        if g in STUDY_GROUPS or g == VHH_GROUP:
            continue
        s, l = _halve({r["lineage_key"] for r in recs})
        selection[g] = s
        loco[g] = l

    return selection, loco, study_alloc


@dataclass
class AuditReport:
    decision: str
    role_overlap: dict                     # {"lineage": n, "donor": n, "study": n}
    counts_by_role: dict                   # role -> group -> count
    counts_by_locus: dict                  # role -> category -> count
    counts_by_study: dict                  # role -> #studies
    counts_by_donor: dict                  # role -> #donors
    floors_ok: dict
    locus_balance_37_37: dict
    locus_diversity: dict                   # cat -> role -> {donors, studies, ok}
    powered_real_igh: dict
    policy_hash_reproduced: bool
    source_checksums: dict
    failures: list = field(default_factory=list)


def _role_of_lineage(selection, loco):
    role = {}
    for g, s in selection.items():
        for lk in s:
            role[lk] = "selection"
    for g, s in loco.items():
        for lk in s:
            role.setdefault(lk, None)
            role[lk] = "loco" if role[lk] is None else "BOTH"
    return role


def audit_stage1(*, records, selection, loco, study_alloc, source_checksums, expected_policy_hash=None,
                 required_groups=REQUIRED_GROUPS):
    failures = []

    # 0. all binding groups must be present (a partial cohort STOPs, never falsely GOes).
    present_groups = set(selection) | set(loco)
    missing = set(required_groups) - present_groups
    if missing:
        failures.append(f"missing_groups {sorted(missing)}")

    # 1. ZERO role overlap — lineage, donor, study.
    lineage_overlap = 0
    for g in set(selection) | set(loco):
        lineage_overlap += len(selection.get(g, set()) & loco.get(g, set()))
    # donor/study overlap: map each lineage to its donor/study. Structural records share their parent's
    # lineage_key but carry no study/donor; if one shadowed the parent here, a positive_real-only study
    # would become invisible to the overlap check. Prefer the study-bearing record for each lineage_key.
    lk_meta = {}
    for r in records:
        lk = r["lineage_key"]
        prev = lk_meta.get(lk)
        if prev is None or (r.get("study") and not prev.get("study")):
            lk_meta[lk] = r
    sel_lk = set().union(*selection.values()) if selection else set()
    loc_lk = set().union(*loco.values()) if loco else set()

    def _donors(lks):
        return {(lk_meta[lk].get("study"), lk_meta[lk].get("donor"))
                for lk in lks if lk in lk_meta and lk_meta[lk].get("donor")}

    def _studies(lks):
        return {canonical_study(lk_meta[lk].get("repository", "oas"), lk_meta[lk].get("study"))
                for lk in lks if lk in lk_meta and lk_meta[lk].get("study")}

    donor_overlap = len(_donors(sel_lk) & _donors(loc_lk))
    study_overlap = len(_studies(sel_lk) & _studies(loc_lk))
    role_overlap = {"lineage": lineage_overlap, "donor": donor_overlap, "study": study_overlap}
    if any(role_overlap.values()):
        failures.append(f"role_overlap {role_overlap}")

    # 2/3. counts by role / locus / study / donor — computed from the POST-DEDUP `records` assigned to
    # each role (owner directive: recheck floors after global dedup). Counting records (not the pre-dedup
    # lineage-key sets) also gives an accurate per-locus census: structural records share their parent's
    # lineage_key, so a set-based count would both double-count and mislabel them.
    def _role_of(rec):
        g, lk = rec.get("group"), rec.get("lineage_key")
        if lk in selection.get(g, set()):
            return "selection"
        if lk in loco.get(g, set()):
            return "loco"
        return None

    role_group_lineages = {"selection": {}, "loco": {}}   # role -> group -> set(lineage_key), post-dedup
    counts_by_locus = {"selection": {}, "loco": {}}
    for rec in records:
        role = _role_of(rec)
        if role is None:
            continue
        g, lk, cat = rec.get("group"), rec.get("lineage_key"), rec.get("category", "?")
        role_group_lineages[role].setdefault(g, set()).add(lk)
        counts_by_locus[role][cat] = counts_by_locus[role].get(cat, 0) + 1
    counts_by_role = {role: {g: len(lks) for g, lks in gm.items()}
                      for role, gm in role_group_lineages.items()}
    counts_by_study = {"selection": len(_studies(sel_lk)), "loco": len(_studies(loc_lk))}
    counts_by_donor = {"selection": len(_donors(sel_lk)), "loco": len(_donors(loc_lk))}

    # 4. floors per group per role (guardband floors), on the post-dedup counts.
    floors_ok = {}
    for g in set(selection) | set(loco):
        floor = GUARDBAND_FLOOR[GROUP_GATE_KIND.get(g, "gate_10pct")]
        floors_ok[g] = {"selection": counts_by_role["selection"].get(g, 0) >= floor,
                        "loco": counts_by_role["loco"].get(g, 0) >= floor}
        if not (floors_ok[g]["selection"] and floors_ok[g]["loco"]):
            failures.append(f"floor {g} {floors_ok[g]}")

    # 5. IGK/IGL 37/37 balance per subset + powered real-IGH.
    locus_balance = {}
    for g, loci in GROUP_LOCI.items():
        if g not in (set(selection) | set(loco)):
            continue
        ok = True
        for role, cbl in (("selection", counts_by_locus["selection"]), ("loco", counts_by_locus["loco"])):
            for loc_ in loci:
                if cbl.get(loc_, 0) < 37:
                    ok = False
        locus_balance[g] = ok
        if not ok:
            failures.append(f"locus_balance_37_37 {g}")
    powered_real_igh = {"selection": counts_by_role["selection"].get("positive_real", 0) >= 36,
                        "loco": counts_by_role["loco"].get("positive_real", 0) >= 36}
    if "positive_real" in (set(selection) | set(loco)) and not all(powered_real_igh.values()):
        failures.append(f"real_igh_underpowered {powered_real_igh}")

    # 5b. per-role per-locus donor/study DIVERSITY (no validation slice may rest on one donor/study).
    rl_donors = {"selection": {}, "loco": {}}
    rl_studies = {"selection": {}, "loco": {}}
    for rec in records:
        cat = rec.get("category")
        if cat not in LOCUS_DIVERSITY or not rec.get("study"):
            continue
        role = _role_of(rec)
        if role is None:
            continue
        cs = canonical_study(rec.get("repository", "oas"), rec.get("study"))
        rl_donors[role].setdefault(cat, set()).add((cs, rec.get("donor")))
        rl_studies[role].setdefault(cat, set()).add(cs)
    present_cats = set(counts_by_locus["selection"]) | set(counts_by_locus["loco"])
    locus_diversity = {}
    for cat, tgt in LOCUS_DIVERSITY.items():
        if cat not in present_cats:
            continue
        locus_diversity[cat] = {}
        for role in ("selection", "loco"):
            nd = len(rl_donors[role].get(cat, set()))
            ns = len(rl_studies[role].get(cat, set()))
            ok = nd >= tgt["donors"] and ns >= tgt["studies"]
            locus_diversity[cat][role] = {"donors": nd, "studies": ns, "ok": ok}
            if not ok:
                failures.append(f"locus_diversity {cat} {role} donors={nd}<{tgt['donors']} or studies={ns}<{tgt['studies']}")

    # 6. frozen allocator policy hash reproduced exactly.
    policy_ok = True
    if study_alloc is not None:
        if expected_policy_hash is not None:
            policy_ok = study_alloc.policy_hash == expected_policy_hash
        # reproducibility: re-run must yield the same hash
        study_records = [r for r in records if r.get("group") in STUDY_GROUPS]
        if study_records:
            policy_ok = policy_ok and assign_studies(study_records).policy_hash == study_alloc.policy_hash
    if not policy_ok:
        failures.append("policy_hash_not_reproduced")

    decision = "GO" if not failures else "STOP_STAGE1_AUDIT:" + ";".join(failures)
    return AuditReport(
        decision=decision, role_overlap=role_overlap, counts_by_role=counts_by_role,
        counts_by_locus=counts_by_locus, counts_by_study=counts_by_study, counts_by_donor=counts_by_donor,
        floors_ok=floors_ok, locus_balance_37_37=locus_balance, locus_diversity=locus_diversity,
        powered_real_igh=powered_real_igh, policy_hash_reproduced=policy_ok,
        source_checksums=source_checksums, failures=failures)

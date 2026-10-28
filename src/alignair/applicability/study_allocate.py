"""Study/donor-level split-barrier allocation for donor-based sources (OAS, VDJServer).

Binding behavior (owner-specified):
  * Assign STUDY first, donor second, lineage last — never split a study or donor across roles.
  * Canonical identity includes repository + study + donor; known aliases are collapsed and audited.
  * OAS heavy and light records from the same study inherit the same role.
  * Optimize deterministically for per-role per-locus floors + 37/37 balance; oversample rather than split.
  * Freeze policy + seed + study-list hash + tie-break BEFORE hydration results are scored (this module
    runs on identities only — no model outputs — so the allocation is a pure function of the frozen inputs).

Global sequence dedup happens AFTER assignment (in the Stage-1 gate), then floors are re-checked.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

# Per-role (= per-subset) per-locus guardband floors: 37 for the balanced 5%-gate loci, 36 for positives.
PER_ROLE_TARGET = {"igk": 37, "igl": 37, "tra": 37, "trb": 37, "pos_igh_repertoire": 36}
# Constituent loci that must each clear the per-locus floor within a combined group.
GROUP_LOCI = {"igk_igl": ["igk", "igl"], "tcr": ["tra", "trb"]}
# Per-role per-locus donor/study diversity targets — the allocator steers toward these (and the Stage-1
# audit hard-gates them) so no validation locus slice rests on a single donor/study. Light chains (paired
# repertoires, many donors) require >=3 donors / >=2 studies; TCR (10x, four donor-datasets) >=2 / >=2.
DIVERSITY_TARGET = {"igk": {"donors": 3, "studies": 2}, "igl": {"donors": 3, "studies": 2},
                    "tra": {"donors": 2, "studies": 2}, "trb": {"donors": 2, "studies": 2}}


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def canonical_study(repository: str, study: str) -> str:
    """repository::surname_year for 'Author et al., YYYY' studies — collapses aliases like
    'Bernat_2019_1/2' and 'Bernat et al., 2019'. For yearless studies (e.g. 10x datasets, which share a
    leading token like '10x_vdj_...') fall back to repository::full-slug so distinct datasets are never
    merged by a shared surname token."""
    repo = (repository or "oas").lower()
    s = (study or "").lower().strip()
    yr = re.search(r"(?:19|20)\d\d", s)
    if yr:
        surname = re.split(r"[ _,]", s)[0] if s else ""
        return f"{repo}::{surname}_{yr.group()}"
    slug = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return f"{repo}::{slug}"


@dataclass(frozen=True)
class StudyAllocation:
    role_of_study: dict       # canonical_study -> "selection" | "loco_challenge"
    selection: dict           # group -> set[lineage_key]
    loco_challenge: dict      # group -> set[lineage_key]
    per_role_category: dict   # role -> category -> distinct-lineage count
    floors_ok: dict           # group -> {role: bool}  (group total >= combined floor)
    locus_min_ok: dict        # group -> {role: bool}  (every constituent locus >= 37)
    alias_map: dict           # canonical_study -> sorted[raw study labels] (audit)
    policy_hash: str          # frozen policy + seed + study-list + tie-break


def assign_studies(records, *, targets=PER_ROLE_TARGET, diversity=DIVERSITY_TARGET) -> StudyAllocation:
    by_study: dict = {}
    alias: dict = {}
    for r in records:
        cs = canonical_study(r.get("repository", "oas"), r.get("study", ""))
        by_study.setdefault(cs, []).append(r)
        alias.setdefault(cs, set()).add(r.get("study", ""))

    # per-study distinct-lineage counts by category + the donor set each study contributes per category
    study_counts: dict = {}
    study_donors: dict = {}
    for cs, recs in by_study.items():
        seen, cc, dd = set(), {}, {}
        for r in recs:
            key = (r["category"], r["lineage_key"])
            if key in seen:
                continue
            seen.add(key)
            cc[r["category"]] = cc.get(r["category"], 0) + 1
            if r.get("donor"):
                dd.setdefault(r["category"], set()).add((cs, r["donor"]))
        study_counts[cs] = cc
        study_donors[cs] = dd

    def deficit(rc):
        return sum(max(0, targets.get(c, 0) - rc.get(c, 0)) for c in targets)

    def div_deficit(dn, stu):
        """Unmet donor+study diversity across the balanced loci for one role's (donors, studies) sets."""
        return sum(max(0, t["donors"] - len(dn.get(c, set()))) + max(0, t["studies"] - len(stu.get(c, set())))
                   for c, t in diversity.items())

    def total(rc):
        return sum(rc.values())

    role_counts = {"selection": {}, "loco_challenge": {}}
    role_donors = {"selection": {}, "loco_challenge": {}}    # role -> cat -> set[(cs, donor)]
    role_studies = {"selection": {}, "loco_challenge": {}}   # role -> cat -> set[cs]
    role_of: dict = {}
    # Deterministic greedy in frozen sha256 order; assign each study to the role minimizing, in priority:
    # (1) total count deficit (floors), (2) total donor/study diversity deficit, (3) count imbalance; tie
    # -> selection. (2) is what steers a whole light study into whichever role still lacks IGL/IGK donors.
    for cs in sorted(by_study, key=_sha):
        sc = study_counts[cs]
        sd = study_donors[cs]
        best = None
        for role in ("selection", "loco_challenge"):
            other = "loco_challenge" if role == "selection" else "selection"
            trial = dict(role_counts[role])
            for c, n in sc.items():
                trial[c] = trial.get(c, 0) + n
            t_dn = {c: set(s) for c, s in role_donors[role].items()}
            t_stu = {c: set(s) for c, s in role_studies[role].items()}
            for c in sc:
                t_dn.setdefault(c, set()).update(sd.get(c, set()))
                t_stu.setdefault(c, set()).add(cs)
            score = (deficit(trial) + deficit(role_counts[other]),
                     div_deficit(t_dn, t_stu) + div_deficit(role_donors[other], role_studies[other]),
                     abs(total(trial) - total(role_counts[other])),
                     0 if role == "selection" else 1)
            if best is None or score < best[0]:
                best = (score, role, trial, t_dn, t_stu)
        _, role, trial, t_dn, t_stu = best
        role_counts[role] = trial
        role_donors[role] = t_dn
        role_studies[role] = t_stu
        role_of[cs] = role

    selection: dict = {}
    loco: dict = {}
    for cs, recs in by_study.items():
        tgt = selection if role_of[cs] == "selection" else loco
        for r in recs:
            tgt.setdefault(r["group"], set()).add(r["lineage_key"])

    # floors + per-locus balance (from distinct-lineage counts by category per role)
    per_role_cat = {"selection": role_counts["selection"], "loco_challenge": role_counts["loco_challenge"]}
    groups = {r["group"] for r in records}
    floors_ok: dict = {}
    locus_min_ok: dict = {}
    for g in groups:
        loci = GROUP_LOCI.get(g)
        floors_ok[g] = {}
        locus_min_ok[g] = {}
        for role in ("selection", "loco_challenge"):
            role_set = (selection if role == "selection" else loco).get(g, set())
            if loci:
                combined_floor = sum(targets.get(loc_, 0) for loc_ in loci)
                floors_ok[g][role] = len(role_set) >= combined_floor
                locus_min_ok[g][role] = all(per_role_cat[role].get(loc_, 0) >= targets.get(loc_, 0)
                                            for loc_ in loci)
            else:
                # single-category group (e.g. positive_real via pos_igh_repertoire)
                cat_floor = max((targets.get(r["category"], 0) for r in records if r["group"] == g),
                                default=36)
                floors_ok[g][role] = len(role_set) >= cat_floor

    policy = {"rule": "deterministic_greedy_min_deficit_then_diversity_then_balance",
              "unit": "canonical_study (repository::surname_year)",
              "seed": "sha256(canonical_study) ascending",
              "tie_break": "lower_total_deficit, then lower diversity-deficit, then count-imbalance, then selection",
              "targets": targets, "diversity": diversity, "study_list": sorted(by_study)}
    policy_hash = _sha(json.dumps(policy, sort_keys=True))

    return StudyAllocation(
        role_of_study=role_of, selection=selection, loco_challenge=loco,
        per_role_category=per_role_cat, floors_ok=floors_ok, locus_min_ok=locus_min_ok,
        alias_map={cs: sorted(v) for cs, v in alias.items()}, policy_hash=policy_hash)

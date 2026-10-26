"""Feature preparation and target masks for guard training."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch

from alignair.aa.guard.classes import COMPLETENESS_INDEX, DOMAIN_TYPE_INDEX
from alignair.aa.guard.features import FEATURE_NAMES, hmm_features
from alignair.aa.guard.tokenizer import enforce_length_contract
from alignair.errors import DataContractError

COMPLETE_MAP = {"cdr3_only": "too_short", "j_only": "missing_v_anchor", "v_only": "missing_j_anchor",
                "leader_vh": "contaminated", "vh_constant": "contaminated"}


def precompute_features(records, bank, *, bank_checksum, cache_path, mask_groups=frozenset()):
    """Cache features bound to the bank, feature ordering, and excluded profiles.

    Legacy unbound caches or caches from a different bank/fold are rebuilt.
    Malformed JSON is reported rather than overwritten.
    """
    path = Path(cache_path)
    identity = {
        "schema_version": 1, "bank_checksum": bank_checksum,
        "feature_names": list(FEATURE_NAMES), "mask_groups": sorted(mask_groups),
    }
    stored = {}
    if path.exists():
        try:
            stored = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise DataContractError("Invalid guard feature cache JSON.", context={"path": path}) from exc
        if not isinstance(stored, dict):
            raise DataContractError("Guard feature cache must be a mapping.", context={"path": path})
    matched = all(stored.get(key) == value for key, value in identity.items())
    cache = stored.get("vectors", {}) if matched else {}
    if not isinstance(cache, dict):
        raise DataContractError("Guard feature cache vectors must be a mapping.", context={"path": path})
    changed = not matched
    for record in records:
        key = hashlib.sha256(record["aa"].encode()).hexdigest()
        if key not in cache:
            cache[key] = hmm_features(bank, record["aa"], mask_groups=mask_groups).to_vector().tolist()
            changed = True
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**identity, "vectors": cache}))
    return cache


def domain_type(rec):
    g, cat = rec.get("group"), rec.get("category", "")
    if g in ("positive_real", "positive_sim"):
        return DOMAIN_TYPE_INDEX["human_igh_vh"]
    if g == "vhh_nonhuman":
        return DOMAIN_TYPE_INDEX["murine_vh"] if cat in ("murine_vh", "chimeric_murine_variable") \
            else DOMAIN_TYPE_INDEX["vhh"]
    if g == "igsf":
        return DOMAIN_TYPE_INDEX["igsf"]
    if g == "igk_igl":
        return DOMAIN_TYPE_INDEX["human_igk"] if cat == "igk" else DOMAIN_TYPE_INDEX["human_igl"]
    if g == "tcr":
        return DOMAIN_TYPE_INDEX["tra"] if cat == "tra" else DOMAIN_TYPE_INDEX["trb"]
    return -1                                            # structural / other -> no domain-type label


def completeness(rec):
    if rec.get("group") in ("positive_real", "positive_sim"):
        return COMPLETENESS_INDEX["complete"]
    c = COMPLETE_MAP.get(rec.get("category", ""))
    return COMPLETENESS_INDEX[c] if c else -1            # reversed_vh / others -> masked


def make_oe(train, seed, cap_n):
    rng = random.Random(seed)
    oe = []
    for r in rng.sample(train, min(len(train), cap_n * 4)):
        chars = list(r["aa"])
        rng.shuffle(chars)                               # residue-shuffle destroys the domain -> outlier
        oe.append({"id": f"oe::{r['id']}", "aa": "".join(chars), "group": "oe"})
    return oe


def source_setup(train):
    """Category as a source proxy; a domain-type is source-adv ELIGIBLE only if it has >=2 source categories
    (Safeguard 4). Returns (cat2id, eligibledomain_types)."""
    from collections import defaultdict
    cats_by_dt = defaultdict(set)
    for r in train:
        dt = domain_type(r)
        if dt >= 0:
            cats_by_dt[dt].add(r.get("category", ""))
    all_cats = sorted({c for cats in cats_by_dt.values() for c in cats})
    cat2id = {c: i for i, c in enumerate(all_cats)}
    eligible = {dt for dt, cats in cats_by_dt.items() if len(cats) >= 2}
    return cat2id, eligible, len(all_cats)


def build_tensors(records, feat_cache, fusion, tok, bank_checksum, cat2id, eligible_dt, is_oe=False,
                  mask_groups=frozenset(), *, bank):
    """Build tensors with objective-specific masks and an explicitly supplied HMM bank."""
    ids_l, mask_l, hmm_l = [], [], []
    dt_t, dt_m, cp_t, cp_m, app_t, en_out, en_m, src_t, src_m, ct_t, ct_m = ([] for _ in range(11))
    for r in records:
        if not enforce_length_contract(r["aa"]).ok:
            continue
        h = hashlib.sha256(r["aa"].encode()).hexdigest()
        vec = feat_cache.get(h) or hmm_features(bank, r["aa"], mask_groups=mask_groups).to_vector().tolist()
        hmm_l.append(fusion.normalize(vec, feature_names=FEATURE_NAMES, bank_checksum=bank_checksum))
        i, m = tok.encode(r["aa"]); ids_l.append(i); mask_l.append(m)
        if is_oe:
            dt_t.append(0); dt_m.append(0.0); cp_t.append(0); cp_m.append(0.0)
            app_t.append(0.0); en_out.append(1.0); en_m.append(1.0)
            src_t.append(0); src_m.append(0.0); ct_t.append(0); ct_m.append(0.0)
            continue
        dt = domain_type(r); cp = completeness(r)
        dt_t.append(max(dt, 0)); dt_m.append(1.0 if dt >= 0 else 0.0)
        cp_t.append(max(cp, 0)); cp_m.append(1.0 if cp >= 0 else 0.0)
        app_t.append(1.0 if r.get("expected_state") == "supported" else 0.0)
        en_out.append(0.0); en_m.append(1.0 if dt >= 0 else 0.0)      # in-domain samples only
        elig = dt >= 0 and dt in eligible_dt
        src_t.append(cat2id.get(r.get("category", ""), 0)); src_m.append(1.0 if elig else 0.0)
        ct_t.append(max(dt, 0)); ct_m.append(1.0 if dt >= 0 else 0.0)
    if not ids_l:
        raise DataContractError("No records satisfy the guard training length contract.")
    t = lambda x, dt_: torch.tensor(x, dtype=dt_)  # noqa: E731
    return {"ids": torch.stack(ids_l), "mask": torch.stack(mask_l), "hmm": torch.stack(hmm_l),
            "domain_type": t(dt_t, torch.long), "type_m": t(dt_m, torch.float),
            "completeness": t(cp_t, torch.long), "comp_m": t(cp_m, torch.float),
            "applicability": t(app_t, torch.float),
            "energy_out": t(en_out, torch.float), "energy_m": t(en_m, torch.float),
            "source": t(src_t, torch.long), "source_m": t(src_m, torch.float),
            "contrast": t(ct_t, torch.long), "contrast_m": t(ct_m, torch.float)}

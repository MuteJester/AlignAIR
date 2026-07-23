"""Promotion confirmation: incumbent vs winner ONLY, larger untouched cohort, paired per-read analysis.

Implements the FROZEN predeclaration in docs/superpowers/artifacts/2026-07-22-aa-convergence/
promotion_criteria.md. No training; no intermediate checkpoints; no Phase-3 data. Writes confirmation.json +
confirmation_result.md and prints PROMOTE / HOLD per the 5 predeclared criteria.
"""
import argparse
import json

import numpy as np
import torch
import GenAIRR as ga

from alignair.aa.strategic import strategic_setup
from alignair.aa.gym_bridge import aa_usable
from alignair.aa.infer import class_probs
from alignair.aa.conformal import Conformal
from alignair.aa.convergence.evaluator import build_cohort, _first
from alignair.aa.convergence.metrics import macro_accuracy, per_frequency_bins
from alignair.aa.convergence.paired import (
    paired_bootstrap_diff_ci, mcnemar, class_stratified_bootstrap_macro_diff)
from alignair_benchmark import default_igh_spec, stream_benchmark
from alignair.reference.reference_set import ReferenceSet

INC = "experiments/aa/strategic/model_stage1_PASS.pt"
WIN = "experiments/aa/strategic/model_cont_step318000_modelonly.pt"
ART = "docs/superpowers/artifacts/2026-07-22-aa-convergence"


def _load(path, device):
    model, cfg, cmaps, meta = strategic_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.to(device).eval()
    return model, cfg, cmaps, meta


def _empirical_class_freq(dc, ref, class_map, n_reads, seed):
    """Corrected freq prior: empirical V-class frequency from a frozen stream sample (NOT uniform gene_use)."""
    vclass = class_map.allele_class
    cnt = np.zeros(class_map.n_classes())
    n = 0
    for c in stream_benchmark(default_igh_spec(n_per_stratum=400, seed=seed), dataconfig=dc, reference_set=ref):
        r = c.record
        if c.orientation_id != 0 or not aa_usable(r):
            continue
        call = _first(r["v_call"])
        if call in vclass:
            cnt[vclass[call]] += 1
            n += 1
        if n >= n_reads:
            break
    return cnt / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    dev = args.device
    dc = ga.HUMAN_IGH_OGRDB
    ref = ReferenceSet.from_dataconfigs(dc)

    inc, cfg, cmaps, meta = _load(INC, dev)
    win, _, _, _ = _load(WIN, dev)
    vclass, jclass = cmaps["V"].allele_class, cmaps["J"].allele_class
    d_index = meta["d_index"]

    # ---- confirmation cohort (larger, NEW seed) ----
    cohort, fp = build_cohort("HUMAN_IGH_OGRDB", args.n, seed=770077)
    freq = _empirical_class_freq(dc, ref, cmaps["V"], n_reads=8000, seed=880088)
    # frozen bins from the prior (rare/mid/common by class frequency)
    nz = freq[freq > 0]
    e1, e2 = float(np.quantile(nz, 1 / 3)), float(np.quantile(nz, 2 / 3))
    freq_edges = (0.0, e1, e2, 1.0)

    def preds(model, rows, key):
        p = class_probs(model, [r["aa"] for r in rows], cfg, dev, key)
        return p, p.argmax(1)

    report = {"cohort_fingerprint": fp, "n_per_stratum": args.n, "freq_edges": freq_edges, "strata": {}}
    # ---- per-stratum paired accuracy (V) ----
    all_pass_stratum = True
    for name, rows in cohort["by_stratum"].items():
        if not rows:
            continue
        y = np.array([vclass[_first(r["v_call"])] for r in rows])
        _, pa = preds(inc, rows, "v_class_logits")
        _, pb = preds(win, rows, "v_class_logits")
        a_ok, b_ok = (pa == y), (pb == y)
        bc = paired_bootstrap_diff_ci(a_ok, b_ok, seed=0)
        mc = mcnemar(a_ok, b_ok)
        report["strata"][name] = {
            "n": len(rows), "inc_v_acc": float(a_ok.mean()), "win_v_acc": float(b_ok.mean()),
            "diff": bc["diff"], "ci": [bc["lo"], bc["hi"]],
            "mcnemar_p": mc["p_value"], "mcnemar_net": mc["net"]}
        # criterion 4: no stratum loses > 1pt (point) and no McNemar-significant regression > 1pt
        if b_ok.mean() < a_ok.mean() - 0.01 and (mc["p_value"] < 0.05 and mc["net"] < 0):
            all_pass_stratum = False

    clean = cohort["by_stratum"]["clean"]
    yv = np.array([vclass[_first(r["v_call"])] for r in clean])
    yj = np.array([jclass[_first(r["j_call"])] for r in clean])
    _, pav = preds(inc, clean, "v_class_logits"); _, pbv = preds(win, clean, "v_class_logits")
    _, paj = preds(inc, clean, "j_class_logits"); _, pbj = preds(win, clean, "j_class_logits")

    # ---- V macro (class-stratified bootstrap) ----
    vmac = class_stratified_bootstrap_macro_diff(yv, pav == yv, pbv == yv, seed=0)
    # ---- freq bins (corrected empirical prior) ----
    fv = np.array([freq[c] for c in yv])
    fb_inc = per_frequency_bins(yv, pav, fv, freq_edges)
    fb_win = per_frequency_bins(yv, pbv, fv, freq_edges)

    # ---- D (allele macro via class-strat bootstrap; gene acc via paired bootstrap) ----
    d_rows = [r for r in clean if _first(r.get("d_call", "") or "") in d_index]
    yd = np.array([d_index[_first(r["d_call"])] for r in d_rows])
    _, pad = preds(inc, d_rows, "d_class_logits"); _, pbd = preds(win, d_rows, "d_class_logits")
    d_name = {i: n for n, i in d_index.items()}
    gene = lambda i: d_name[i].split("*")[0]
    ygene = np.array([gene(c) for c in yd])
    dgene_a = np.array([gene(c) == g for c, g in zip(pad, ygene)])
    dgene_b = np.array([gene(c) == g for c, g in zip(pbd, ygene)])
    d_allele_mac = class_stratified_bootstrap_macro_diff(yd, pad == yd, pbd == yd, seed=0)
    d_gene = paired_bootstrap_diff_ci(dgene_a, dgene_b, seed=0)

    report["clean"] = {
        "inc_clean_v": float((pav == yv).mean()), "win_clean_v": float((pbv == yv).mean()),
        "inc_clean_j": float((paj == yj).mean()), "win_clean_j": float((pbj == yj).mean()),
        "v_macro_inc": macro_accuracy(yv, pav), "v_macro_win": macro_accuracy(yv, pbv),
        "v_macro_diff_ci": [vmac["diff"], vmac["lo"], vmac["hi"]],
        "d_allele_macro_inc": macro_accuracy(yd, pad), "d_allele_macro_win": macro_accuracy(yd, pbd),
        "d_allele_macro_diff_ci": [d_allele_mac["diff"], d_allele_mac["lo"], d_allele_mac["hi"]],
        "d_gene_inc": float(dgene_a.mean()), "d_gene_win": float(dgene_b.mean()),
        "d_gene_diff_ci": [d_gene["diff"], d_gene["lo"], d_gene["hi"]],
        "freq_bins_inc": fb_inc, "freq_bins_win": fb_win}

    # ---- conformal on SEPARATE calibration + test cohorts ----
    cal_c, _ = build_cohort("HUMAN_IGH_OGRDB", 1500, seed=990099)
    tst_c, _ = build_cohort("HUMAN_IGH_OGRDB", 1500, seed=112211)
    def cov(model):
        cr, tr = cal_c["by_stratum"]["clean"], tst_c["by_stratum"]["clean"]
        pc = class_probs(model, [r["aa"] for r in cr], cfg, dev, "v_class_logits")
        yc = np.array([vclass[_first(r["v_call"])] for r in cr])
        conf = Conformal(alpha=0.05, seed=0).fit(pc, yc)
        pt = class_probs(model, [r["aa"] for r in tr], cfg, dev, "v_class_logits")
        yt = np.array([vclass[_first(r["v_call"])] for r in tr])
        covered = [yt[i] in conf.predict_set(pt[i]) for i in range(len(yt))]
        sizes = [len(conf.predict_set(pt[i])) for i in range(len(yt))]
        return float(np.mean(covered)), float(np.mean(sizes))
    inc_cov, inc_set = cov(inc); win_cov, win_set = cov(win)
    report["conformal"] = {"inc_coverage": inc_cov, "inc_set_mean": inc_set,
                           "win_coverage": win_cov, "win_set_mean": win_set}

    # ---- apply the 5 FROZEN promotion criteria ----
    cl = report["clean"]
    c1 = cl["win_clean_v"] >= 0.980 and cl["win_clean_j"] >= 0.988
    dam, dgc = cl["d_allele_macro_diff_ci"], cl["d_gene_diff_ci"]
    da_pt = cl["d_allele_macro_win"] > cl["d_allele_macro_inc"]
    dg_pt = cl["d_gene_win"] > cl["d_gene_inc"]
    at_least_one_sig = (dam[1] > 0) or (dgc[1] > 0)
    neither_sig_reg = (dam[2] >= 0) and (dgc[2] >= 0)
    c2 = da_pt and dg_pt and at_least_one_sig and neither_sig_reg
    def frontier_ok(nm):
        s = report["strata"].get(nm)
        return s and s["win_v_acc"] >= s["inc_v_acc"] - 0.01 and s["ci"][0] > -0.01
    c3 = frontier_ok("moderate_full") and frontier_ok("v_indel_multi")
    c4 = all_pass_stratum and all(s["win_v_acc"] >= s["inc_v_acc"] - 0.01 for s in report["strata"].values())
    c5 = win_cov >= 0.94
    verdict = {"c1_gates": c1, "c2_D_gains": c2, "c3_frontier_no_regress": c3,
               "c4_no_stratum_loses_1pt": c4, "c5_coverage_valid": c5}
    promote = all(verdict.values())
    report["criteria"] = verdict
    report["decision"] = "PROMOTE" if promote else "HOLD"

    json.dump(report, open(f"{ART}/confirmation.json", "w"), indent=2, sort_keys=True)
    print("=== PROMOTION CONFIRMATION (incumbent vs winner) ===")
    print(f"clean V inc {cl['inc_clean_v']:.4f} -> win {cl['win_clean_v']:.4f} | clean J {cl['inc_clean_j']:.4f} -> {cl['win_clean_j']:.4f}")
    print(f"D-allele-macro {cl['d_allele_macro_inc']:.4f} -> {cl['d_allele_macro_win']:.4f}  diffCI {dam}")
    print(f"D-gene         {cl['d_gene_inc']:.4f} -> {cl['d_gene_win']:.4f}  diffCI {dgc}")
    for nm in ("moderate_full", "v_indel_multi"):
        s = report["strata"].get(nm, {})
        print(f"{nm:15s} {s.get('inc_v_acc',0):.4f} -> {s.get('win_v_acc',0):.4f}  diffCI [{s.get('ci',[None,None])[0]},{s.get('ci',[None,None])[1]}]  mcnemar_p {s.get('mcnemar_p')}")
    print(f"conformal V coverage: inc {inc_cov:.4f} -> win {win_cov:.4f} (gate >=0.94)")
    print("criteria:", verdict)
    print("DECISION:", report["decision"])


if __name__ == "__main__":
    main()

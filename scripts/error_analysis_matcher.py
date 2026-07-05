"""Study the cases where IgBLAST is RIGHT and AIRRMatch (hybrid) is WRONG, to find the
source of each weakness. Decomposes V losses into:
  - seed-retrieval failure  (truth V not even in the k-mer top-k)  -> fix seed
  - fine-pick failure        (truth in shortlist but align picked a sibling) -> fix V scorer
  - sibling vs different-gene, and SHM correlation of the losing reads
and reports D/J (neural) correctness by stratum (training-distribution relevant).
"""
import argparse
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import gen_records, run_igblast  # noqa: E402

import GenAIRR.data as gd  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.matcher.config import MatcherConfig  # noqa: E402
from alignair.matcher.model import AIRRMatch  # noqa: E402
from alignair.matcher import tokenizer as tk  # noqa: E402
from alignair.align import get_aligner  # noqa: E402

STRATA = [("clean", 0.0, None, None), ("hard", 1.0, None, None),
          ("heavy_shm.25", 1.0, None, {"mutation_rate": 0.25}),
          ("fragment_120", 1.0, 120, None)]


def _set(v):
    return {x for x in str(v or "").split(",") if x}


def gene_of(a):
    return a.split("*")[0] if a else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/airrmatch_diverse.pt")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ck = torch.load(a.model, map_location=a.device)
    cfg = MatcherConfig(**ck["config"]); m = AIRRMatch(cfg).to(a.device).eval(); m.load_state_dict(ck["model"])
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    cached = m.encode_reference(ref)
    aligner = get_aligner()
    Vseed = cached["seed"]["V"]; Vname2g = cached["groups"]["V"].name_to_group
    torch.set_grad_enabled(False)

    for name, p, crop, ov in STRATA:
        overrides = {"invert_d_prob": 0.0}
        if ov:
            overrides.update(ov)
        recs = gen_records(p, a.n, 123, crop, overrides=overrides)
        igb = run_igblast(recs)
        print(f"\n===== [{name}] n={a.n} =====")

        # per-gene tallies
        cat = {g: Counter() for g in "vdj"}
        v_loss = Counter(); v_loss_shm = []; all_shm = []
        for rec, row in zip(recs, igb):
            seq = str(rec["sequence"]).upper()
            pred = m.predict_seeded(cached, seq, topk=a.topk, fine="hybrid", aligner=aligner)
            shm = rec.get("n_mutations")
            if shm is not None:
                all_shm.append(shm)
            # recompute the chosen strand + V shortlist (mirror predict_seeded) for retrieval check
            read = seq if Vseed.best_hits(seq) >= Vseed.best_hits(tk.revcomp(seq)) else tk.revcomp(seq)
            vsl = set(Vseed.shortlist(read, a.topk)[0])

            for g in "vdj":
                ts = _set(rec.get(f"{g}_call"))
                if not ts:
                    continue
                igb_ok = bool(_set((row or {}).get(f"{g}_call")) & ts)
                us = set(pred.get(g.upper(), {}).get("names", []))
                us_ok = bool(us & ts)
                key = ("igb+" if igb_ok else "igb-") + ("us+" if us_ok else "us-")
                cat[g][key] += 1
                if g == "v" and igb_ok and not us_ok:                # the LOSS cases to study
                    truth_groups = {Vname2g[x] for x in ts if x in Vname2g}
                    retrieved = bool(truth_groups & vsl)
                    our_gene = gene_of(next(iter(us), None))
                    sibling = our_gene in {gene_of(x) for x in ts}
                    v_loss["seed_retrieved_but_mispick" if retrieved else "seed_MISSED_truth"] += 1
                    v_loss["wrong=sibling" if sibling else "wrong=diff_gene"] += 1
                    if shm is not None:
                        v_loss_shm.append(shm)

        for g in "vdj":
            c = cat[g]; tot = sum(c.values())
            both_wrong = c["igb-us-"]; only_us_wrong = c["igb+us-"]; only_igb_wrong = c["igb-us+"]
            print(f"  {g.upper()}: both_right={c['igb+us+']}/{tot}  WE-LOSE(igb+,us-)={only_us_wrong}  "
                  f"WE-WIN(igb-,us+)={only_igb_wrong}  both_wrong={both_wrong}")
        if v_loss:
            print(f"  V-loss source: {dict(v_loss)}")
            if v_loss_shm and all_shm:
                print(f"  V-loss SHM: mean n_mutations of loss reads={sum(v_loss_shm)/len(v_loss_shm):.1f} "
                      f"vs all reads={sum(all_shm)/len(all_shm):.1f}")


if __name__ == "__main__":
    main()

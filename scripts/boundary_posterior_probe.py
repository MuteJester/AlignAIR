"""Diagnose the J-boundary jitter for the REGION-TAGGER decode (scaled_long has no query
decoder). decode_boundaries uses min/max of argmax==gene positions -> fragile to a single
mislabeled position and to a soft crossover. We compare, against GenAIRR truth:
  - HARD  : current decode (min position for start, max+1 for end of argmax==gene)
  - RUN   : start/end of the LONGEST CONTIGUOUS run of argmax==gene (outlier-robust)
  - SOFT  : the 0.5-crossover of the region probability (sub-position via linear interp)
and report how sharp the crossover is (prob step at the boundary, entropy)."""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.data.tokenizer import pad_tokenize  # noqa: E402
from alignair.nn.heads.region import REGION_INDEX  # noqa: E402

FULL = {"clean_full", "moderate_full", "hard_full", "high_shm", "paired_end",
        "productive_only_clean", "high_indel"}


def longest_run(isgene):
    """(start, end_exclusive) of the longest contiguous True run; (-1,-1) if none."""
    best_s = best_e = -1; best = 0; i = 0; n = len(isgene)
    while i < n:
        if isgene[i]:
            j = i
            while j < n and isgene[j]:
                j += 1
            if j - i > best:
                best, best_s, best_e = j - i, i, j
            i = j
        else:
            i += 1
    return best_s, best_e


def cross_up(prob, idx):
    """sub-position where prob crosses 0.5 upward just before idx (linear interp)."""
    if idx <= 0:
        return float(idx)
    a, b = prob[idx - 1], prob[idx]
    if b == a:
        return float(idx)
    return (idx - 1) + (0.5 - a) / (b - a)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ck = torch.load(".private/models/scaled_long.pt", map_location=device)
    cfg = DNAlignAIRConfig(**ck["config"]); m = DNAlignAIR(cfg).to(device); m.load_state_dict(ck["model"]); m.eval()
    ref_emb = m.encode_reference(rs)
    JID, VID = REGION_INDEX["J"], REGION_INDEX["V"]

    seqs, truth = [], []
    for l in open("experiments/headtohead/cases.jsonl"):
        c = json.loads(l); r = c["record"]
        if c["stratum"] not in FULL or c.get("orientation_id", 0) != 0:
            continue
        if "v_sequence_end" not in r or "j_sequence_start" not in r:
            continue
        seqs.append(c["sequence"]); truth.append((r["v_sequence_end"], r["j_sequence_start"]))
        if len(seqs) >= 600:
            break

    A = {(g, d): [] for g in ("v_end", "j_start") for d in ("hard", "run", "soft")}
    step = {"v_end": [], "j_start": []}
    with torch.no_grad():
        for s in range(0, len(seqs), 64):
            chunk = seqs[s:s + 64]; tr = truth[s:s + 64]
            tokens, mask = pad_tokenize(chunk); tokens, mask = tokens.to(device), mask.to(device)
            out = m(tokens, mask, ref_emb)
            P = F.softmax(out["region_logits"], dim=-1).cpu().numpy()  # (B,L,R)
            lab = P.argmax(-1)
            ml = mask.cpu().numpy().astype(bool)
            for i, (tve, tjs) in enumerate(tr):
                L = int(ml[i].sum())
                pj = P[i, :L, JID]; pv = P[i, :L, VID]; lb = lab[i, :L]
                # ---- J start ----
                isj = lb == JID
                if isj.any():
                    hard = int(np.argmax(isj))                       # min position (current)
                    rs_, _ = longest_run(isj); run = rs_
                    soft = cross_up(pj, hard)
                    A[("j_start", "hard")].append(abs(hard - tjs))
                    A[("j_start", "run")].append(abs(run - tjs))
                    A[("j_start", "soft")].append(abs(round(soft) - tjs))
                    step["j_start"].append(float(pj[hard] - pj[max(hard - 1, 0)]))
                # ---- V end ----
                isv = lb == VID
                if isv.any():
                    hard = int(L - 1 - np.argmax(isv[::-1])) + 1      # max position +1 (current)
                    _, re_ = longest_run(isv); run = re_
                    e = min(hard, L - 1)
                    soft = cross_up(1 - pv, e)                        # V prob drops -> (1-pv) rises
                    A[("v_end", "hard")].append(abs(hard - tve))
                    A[("v_end", "run")].append(abs(run - tve))
                    A[("v_end", "soft")].append(abs(round(soft) - tve))
                    step["v_end"].append(float(pv[max(e - 1, 0)] - pv[e]))

    print(f"n={len(seqs)} full-length identity cases\n")
    for g in ("v_end", "j_start"):
        print(f"{g}:")
        for d in ("hard", "run", "soft"):
            e = np.array(A[(g, d)])
            print(f"  {d:5s}: exact={np.mean(e==0):.3f}  <=1={np.mean(e<=1):.3f}  MAE={e.mean():.2f}")
        print(f"  prob step across boundary (sharpness): {np.mean(step[g]):.3f}\n")


if __name__ == "__main__":
    main()

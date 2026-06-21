"""De-risking experiment for the contaminant/abstention gate (critic-prescribed).

Computes the best-candidate V alignment score per read — RAW and LENGTH-NORMALIZED (divided
by predicted read-segment length, the fix for the length-comparability flaw) — across:
  - REAL IGH reads stratified by SHM / fragment length,
  - GenAIRR contaminants (the easy, trivially-separable negative),
  - simulated IGK / IGL out-of-locus reads (the genuine IG-ADJACENT negative).
Then sets tau at the 5th percentile of pooled-real and reports the fraction of each set below
it. The decision: a length-normalized tau must catch contaminants AND IGK/IGL while NOT
rejecting real heavy-SHM / fragment reads.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import gen_records  # noqa: E402

import GenAIRR.data as gdata  # noqa: E402
from alignair.gym.gym import build_experiment  # noqa: E402
from alignair.gym.curriculum import Curriculum  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR, extract_segment, extract_segment_tokens  # noqa: E402
from alignair.data.tokenizer import pad_tokenize  # noqa: E402
from alignair.nn.state_head import state_reliability  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
ck = torch.load(".private/models/scaled_long.pt", map_location=device)
cfg = DNAlignAIRConfig(**ck["config"]); m = DNAlignAIR(cfg).to(device); m.load_state_dict(ck["model"]); m.eval()
ref = m.encode_reference(rs)


@torch.no_grad()
def v_quality(reads, batch=256):
    """Return (raw_best_V_score, seg_len, normalized=raw/seg_len) per read."""
    raw, seglen = [], []
    for s in range(0, len(reads), batch):
        chunk = reads[s:s + batch]
        tok, mask = pad_tokenize(chunk); tok, mask = tok.to(device), mask.to(device)
        out = m(tok, mask, ref); canon = out["canon_tokens"]; pr = out["region_logits"].argmax(-1)
        G = "V"
        seg_tok, seg_mask = extract_segment_tokens(canon, mask, pr, G)
        sl = seg_mask.sum(1).clamp(min=1)                      # predicted V-segment length
        seg_pos = m.germline_encoder.forward_positions(seg_tok, seg_mask)
        seg_state, _ = extract_segment(out["state_logits"], mask, pr, G)
        seg_rel = state_reliability(seg_state)
        kk = min(16, out["match"][G].shape[1])
        topk = out["match"][G].topk(kk, dim=-1).indices
        prp, pmp, ptp = ref[G]["pos_reps"], ref[G]["pos_mask"], ref[G]["pos_tok"]
        B = len(chunk); ridx = torch.arange(B, device=device).repeat_interleave(kk); cidx = topk.reshape(-1)
        sc = m.aligner.alignment_score(seg_pos[ridx], seg_mask[ridx], prp[cidx], pmp[cidx],
                                       seg_tok=seg_tok[ridx], germ_tok=ptp[cidx],
                                       seg_reliability=seg_rel[ridx]).reshape(B, kk)
        raw += sc.max(1).values.cpu().tolist(); seglen += sl.cpu().tolist()
    raw, seglen = np.array(raw), np.array(seglen)
    return raw, seglen, raw / seglen


def light(dc, n, seed, p=0.3):
    exp = build_experiment(dc, Curriculum().params(p))
    return [r["sequence"] for r in exp.stream_records(n=n, seed=seed)]


N = 500
sets = {
    "real_clean":      [r["sequence"] for r in gen_records(0.0, N, 1, None)],
    "real_hard":       [r["sequence"] for r in gen_records(1.0, N, 2, None)],
    "real_highSHM.25": [r["sequence"] for r in gen_records(1.0, N, 3, None, overrides={"mutation_rate": 0.25})],
    "real_frag80":     [r["sequence"] for r in gen_records(1.0, N, 4, 80)],
    "real_frag50":     [r["sequence"] for r in gen_records(1.0, N, 5, 50)],
    "CONTAMINANT":     [r["sequence"] for r in gen_records(0.5, N, 6, None, overrides={"contaminate_prob": 1.0})],
    "IGK_outoflocus":  light(gdata.HUMAN_IGK_OGRDB, N, 7),
    "IGL_outoflocus":  light(gdata.HUMAN_IGL_OGRDB, N, 8),
}

res = {}
print(f"{'set':18s} {'raw_med':>9} {'seglen':>7} {'NORM_med':>9}")
for name, reads in sets.items():
    raw, sl, norm = v_quality(reads)
    res[name] = (raw, sl, norm)
    print(f"{name:18s} {np.median(raw):+9.1f} {np.median(sl):7.0f} {np.median(norm):+9.3f}")

real_keys = [k for k in res if k.startswith("real")]
for label, idx in (("RAW score", 0), ("LENGTH-NORMALIZED score", 2)):
    pooled_real = np.concatenate([res[k][idx] for k in real_keys])
    tau = np.percentile(pooled_real, 5)                        # 5% FPR on pooled real by construction
    print(f"\n=== {label}: tau @ 5th pctile of pooled-real = {tau:+.3f} ===")
    print("  fraction below tau (negatives = detection; reals = FALSE rejection):")
    for name in res:
        frac = float(np.mean(res[name][idx] < tau))
        tag = "<-- detect" if not name.startswith("real") else ("<-- FALSE-REJECT" if frac > 0.10 else "")
        print(f"    {name:18s} {frac*100:5.1f}%  {tag}")

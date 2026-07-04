import torch

_NEG = -1e4


def masked_maxsim(read_emb, read_mask, cand_emb, cand_sal, cand_mask):
    """Salience-weighted MaxSim of each candidate against one read.

    read_emb [Ls,d], read_mask [Ls] bool; cand_emb [C,Lc,d], cand_sal [C,Lc],
    cand_mask [C,Lc] bool. Returns scores [C].
    """
    sim = cand_emb @ read_emb.t()                       # [C,Lc,Ls]
    sim = sim.masked_fill(~read_mask.view(1, 1, -1), _NEG)
    best = sim.max(dim=-1).values                       # [C,Lc]  best read match per cand pos
    best = best.masked_fill(~cand_mask, 0.0)
    w = cand_sal.masked_fill(~cand_mask, 0.0)
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # normalize salience per candidate
    return (w * best).sum(dim=-1)                        # [C]


def pick_strand(v_scores_fwd, v_scores_rc) -> int:
    """0 if forward's best V candidate beats revcomp's, else 1."""
    return 0 if v_scores_fwd.max() >= v_scores_rc.max() else 1

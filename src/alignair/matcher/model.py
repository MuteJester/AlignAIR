import torch
import torch.nn as nn

from .config import MatcherConfig
from .encoder import MatcherEncoder
from .scoring import masked_maxsim, pick_strand
from .loss import selection_ce
from .data import build_sequence_groups, encode_batch, GENES
from . import tokenizer as tk
from . import infer


class AIRRMatch(nn.Module):
    def __init__(self, cfg: MatcherConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = MatcherEncoder(cfg)

    # ---- training ----
    def training_loss(self, batch) -> torch.Tensor:
        r_emb, _ = self.encoder(batch["read_ids"], batch["read_mask"])
        rc_emb, _ = self.encoder(batch["read_rc_ids"], batch["read_rc_mask"])
        losses = []
        for g, gb in batch["genes"].items():
            c_emb, c_sal = self.encoder(gb["cand_ids"], gb["cand_mask"])
            for b in range(r_emb.shape[0]):
                fwd = masked_maxsim(r_emb[b], batch["read_mask"][b], c_emb, c_sal, gb["cand_mask"])
                rc = masked_maxsim(rc_emb[b], batch["read_rc_mask"][b], c_emb, c_sal, gb["cand_mask"])
                scores = torch.maximum(fwd, rc)                # strand-invariant training signal
                if gb["target"][b].any():
                    losses.append(selection_ce(scores, gb["target"][b], self.cfg.tau, self.cfg.margin))
        return torch.stack(losses).mean()

    # ---- inference ----
    @torch.no_grad()
    def predict(self, ref, seq: str):
        present = [g for g in GENES if g in ref.genes]
        groups = {g: build_sequence_groups(ref.gene(g)) for g in present}
        r_ids, r_mask = encode_batch([seq], self.cfg.max_len)
        rc_ids, rc_mask = encode_batch([tk.revcomp(seq)], self.cfg.max_len)
        r_emb, _ = self.encoder(r_ids, r_mask)
        rc_emb, _ = self.encoder(rc_ids, rc_mask)
        cand = {}
        for g in present:
            cids, cmask = encode_batch(groups[g].unique_seqs, self.cfg.max_len)
            cemb, csal = self.encoder(cids, cmask)
            cand[g] = (cemb, csal, cmask)
        # strand pick anchored by V
        vfwd = masked_maxsim(r_emb[0], r_mask[0], *cand["V"])
        vrc = masked_maxsim(rc_emb[0], rc_mask[0], *cand["V"])
        strand = pick_strand(vfwd, vrc)
        emb, mask = (r_emb[0], r_mask[0]) if strand == 0 else (rc_emb[0], rc_mask[0])
        out = {}
        for g in present:
            scores = masked_maxsim(emb, mask, *cand[g])
            gid = int(scores.argmax())
            out[g] = {"names": infer.called_names(gid, groups[g]),
                      "seq": groups[g].unique_seqs[gid]}
        return out

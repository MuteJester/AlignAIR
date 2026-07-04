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

    @property
    def device(self):
        return next(self.parameters()).device

    # ---- training ----
    def _gene_scores(self, batch):
        """One forward pass -> {gene: [(scores[C], target[C]), ...]} with strand-max applied.

        Shared by training_loss (differentiable) and telemetry (detached).
        """
        r_emb, _ = self.encoder(batch["read_ids"], batch["read_mask"])
        rc_emb, _ = self.encoder(batch["read_rc_ids"], batch["read_rc_mask"])
        out = {}
        for g, gb in batch["genes"].items():
            c_emb, c_sal = self.encoder(gb["cand_ids"], gb["cand_mask"])
            recs = []
            for b in range(r_emb.shape[0]):
                fwd = masked_maxsim(r_emb[b], batch["read_mask"][b], c_emb, c_sal, gb["cand_mask"])
                rc = masked_maxsim(rc_emb[b], batch["read_rc_mask"][b], c_emb, c_sal, gb["cand_mask"])
                recs.append((torch.maximum(fwd, rc), gb["target"][b]))
            out[g] = recs
        return out

    def _reduce_loss(self, gene_scores):
        losses = [selection_ce(s, t, self.cfg.tau, self.cfg.margin)
                  for recs in gene_scores.values() for (s, t) in recs if t.any()]
        return torch.stack(losses).mean()

    def training_loss(self, batch) -> torch.Tensor:
        return self._reduce_loss(self._gene_scores(batch))

    # ---- inference ----
    @torch.no_grad()
    def encode_reference(self, ref):
        """Encode a reference set once (cache per-gene candidate embeddings + groups)."""
        dev = self.device
        present = [g for g in GENES if g in ref.genes]
        groups = {g: build_sequence_groups(ref.gene(g)) for g in present}
        cand = {}
        for g in present:
            cids, cmask = encode_batch(groups[g].unique_seqs, self.cfg.max_len)
            cemb, csal = self.encoder(cids.to(dev), cmask.to(dev))
            cand[g] = (cemb, csal, cmask.to(dev))
        return {"present": present, "groups": groups, "cand": cand}

    @torch.no_grad()
    def predict_cached(self, cached, seq: str):
        dev = self.device
        r_ids, r_mask = encode_batch([seq], self.cfg.max_len)
        rc_ids, rc_mask = encode_batch([tk.revcomp(seq)], self.cfg.max_len)
        r_emb, _ = self.encoder(r_ids.to(dev), r_mask.to(dev))
        rc_emb, _ = self.encoder(rc_ids.to(dev), rc_mask.to(dev))
        r_mask, rc_mask = r_mask.to(dev), rc_mask.to(dev)
        cand, groups, present = cached["cand"], cached["groups"], cached["present"]
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

    @torch.no_grad()
    def predict(self, ref, seq: str):
        return self.predict_cached(self.encode_reference(ref), seq)

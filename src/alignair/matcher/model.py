import torch
import torch.nn as nn

from .config import MatcherConfig
from .encoder import MatcherEncoder
from .scoring import masked_maxsim, pick_strand
from .loss import selection_ce
from .data import build_sequence_groups, encode_batch, GENES
from .seed import KmerIndex
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
    def encode_reference(self, ref, seed_k=9):
        """Encode a reference set once (cache candidate embeddings + groups + k-mer seed index)."""
        dev = self.device
        present = [g for g in GENES if g in ref.genes]
        groups = {g: build_sequence_groups(ref.gene(g)) for g in present}
        cand = {}
        for g in present:
            cids, cmask = encode_batch(groups[g].unique_seqs, self.cfg.max_len)
            cemb, csal = self.encoder(cids.to(dev), cmask.to(dev))
            cand[g] = (cemb, csal, cmask.to(dev))
        seed = {g: KmerIndex(groups[g].unique_seqs, k=seed_k) for g in present}
        return {"present": present, "groups": groups, "cand": cand, "seed": seed}

    @torch.no_grad()
    def predict_seeded(self, cached, seq: str, topk: int = 16, fine: str = "hybrid", aligner=None):
        """Fast path: k-mer seed shortlists V to top-k, then fine-score only those.

        fine="neural": MaxSim on the shortlist (encoder needed) — best for D/J.
        fine="align":  classical local alignment on the shortlist (needs `aligner`) — best for V.
        fine="hybrid": align for V, neural for D/J — the accuracy-optimal combination.
        Strand is picked by V k-mer hits; D/J score all candidates (cheap, and short D loses k-mers).
        """
        seed, groups, present, cand = (cached["seed"], cached["groups"],
                                       cached["present"], cached["cand"])
        rc = tk.revcomp(seq)
        read = seq if seed["V"].best_hits(seq) >= seed["V"].best_hits(rc) else rc
        shortlists = {g: (seed[g].shortlist(read, topk)[0] if g == "V" else list(range(groups[g].n)))
                      for g in present}

        def gene_fine(g):
            if fine == "align" or (fine == "hybrid" and g == "V"):
                return "align"
            return "neural"

        r_emb = r_mask = None
        if any(gene_fine(g) == "neural" for g in present):     # encode the read only if any gene needs it
            dev = self.device
            r_ids, rm = encode_batch([read], self.cfg.max_len)
            r_emb, _ = self.encoder(r_ids.to(dev), rm.to(dev))
            r_mask = rm.to(dev)

        out = {}
        for g in present:
            idxs = shortlists[g]
            if gene_fine(g) == "align":
                targets = [groups[g].unique_seqs[i] for i in idxs]
                res = aligner.align_many(read, targets)
                gid = idxs[max(range(len(idxs)),
                               key=lambda j: res[j].score if res[j] is not None else float("-inf"))]
            else:
                ix = torch.tensor(idxs, device=self.device)
                cemb, csal, cmask = cand[g]
                scores = masked_maxsim(r_emb[0], r_mask[0], cemb[ix], csal[ix], cmask[ix])
                gid = idxs[int(scores.argmax())]
            out[g] = {"names": infer.called_names(gid, groups[g]),
                      "seq": groups[g].unique_seqs[gid]}
        return out

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

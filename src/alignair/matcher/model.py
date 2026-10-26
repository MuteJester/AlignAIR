import torch
import torch.nn as nn

from .config import MatcherConfig
from .encoder import MatcherEncoder
from .scoring import BandedAlignmentScorer, masked_maxsim, pick_strand
from .loss import selection_ce, selection_ce_batch
from .data import build_sequence_groups, encode_batch, GENES
from .seed import KmerIndex
from . import tokenizer as tk
from . import infer


class AIRRMatch(nn.Module):
    def __init__(self, cfg: MatcherConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = MatcherEncoder(cfg)
        if cfg.scorer not in {"maxsim", "banded"}:
            raise ValueError("MatcherConfig.scorer must be 'maxsim' or 'banded'")
        self.banded_scorer = (
            BandedAlignmentScorer(
                radius=cfg.band_radius,
                aggregation=cfg.band_aggregation,
                temperature=cfg.band_temperature,
                coverage_weight=cfg.band_coverage_weight,
            )
            if cfg.scorer == "banded" else None
        )

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
            if self.banded_scorer is not None:
                B, C = r_emb.shape[0], c_emb.shape[0]
                z = torch.zeros(B, C, dtype=torch.long, device=r_emb.device)
                fwd_offsets = gb.get("offsets_fwd", z).to(r_emb.device)
                rc_offsets = gb.get("offsets_rc", z).to(r_emb.device)
                fwd = self.banded_scorer(
                    r_emb, batch["read_mask"], c_emb, c_sal, gb["cand_mask"],
                    fwd_offsets, read_ids=batch["read_ids"], cand_ids=gb["cand_ids"])
                rc = self.banded_scorer(
                    rc_emb, batch["read_rc_mask"], c_emb, c_sal, gb["cand_mask"],
                    rc_offsets, read_ids=batch["read_rc_ids"], cand_ids=gb["cand_ids"])
                scores = torch.maximum(fwd, rc)
                out[g] = [(scores[b], gb["target"][b]) for b in range(B)]
                continue
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

    def training_loss(self, batch, n_cand: int = 32) -> torch.Tensor:
        if self.banded_scorer is not None:
            gs = self._gene_scores_sampled(batch, n_cand)
            losses = [selection_ce_batch(s, t, self.cfg.tau, self.cfg.margin)
                      for s, t in gs.values()]
            return torch.stack(losses).mean()
        return self._reduce_loss(self._gene_scores(batch))

    @staticmethod
    def _sample_candidates(target_full, n_cand):
        """target_full [B,C] bool -> (idx [B,K], sub_target [B,K]): all positives + random negs."""
        B, C = target_full.shape
        K = min(n_cand, C)
        rows = []
        for b in range(B):
            pos = target_full[b].nonzero(as_tuple=False).flatten()
            pos = pos[:K]
            neg_pool = (~target_full[b]).nonzero(as_tuple=False).flatten()
            need = K - pos.numel()
            if need > 0 and neg_pool.numel():
                perm = torch.randperm(neg_pool.numel(), device=target_full.device)
                negs = neg_pool[perm[:need]]
                if negs.numel() < need:                          # few negs -> sample w/ replacement
                    extra = neg_pool[torch.randint(neg_pool.numel(), (need - negs.numel(),),
                                                   device=target_full.device)]
                    negs = torch.cat([negs, extra])
            else:
                negs = pos.new_empty(0)
            row = torch.cat([pos, negs])[:K]
            if row.numel() < K:                                  # degenerate tiny reference
                row = torch.cat([row, row[:1].repeat(K - row.numel())])
            rows.append(row)
        idx = torch.stack(rows)                                  # [B,K]
        return idx, torch.gather(target_full, 1, idx)

    def _gene_scores_sampled(self, batch, n_cand):
        """Contrastive banded scoring on a per-read sampled candidate subset -> {g: (s[B,K], t[B,K])}."""
        dev = self.device
        r_emb, _ = self.encoder(batch["read_ids"], batch["read_mask"])
        rc_emb, _ = self.encoder(batch["read_rc_ids"], batch["read_rc_mask"])
        out = {}
        for g, gb in batch["genes"].items():
            idx, sub_target = self._sample_candidates(gb["target"].to(dev), n_cand)
            B, K = idx.shape
            cand_ids = gb["cand_ids"].to(dev)[idx]               # [B,K,Lc]
            cand_mask = gb["cand_mask"].to(dev)[idx]
            Lc = cand_ids.shape[-1]
            c_emb, c_sal = self.encoder(cand_ids.reshape(B * K, Lc), cand_mask.reshape(B * K, Lc))
            c_emb = c_emb.reshape(B, K, Lc, -1)
            c_sal = c_sal.reshape(B, K, Lc)
            z = torch.zeros(B, gb["target"].shape[1], dtype=torch.long, device=dev)
            off_fwd = torch.gather(gb.get("offsets_fwd", z).to(dev), 1, idx)
            off_rc = torch.gather(gb.get("offsets_rc", z).to(dev), 1, idx)
            fwd = self.banded_scorer(r_emb, batch["read_mask"], c_emb, c_sal, cand_mask,
                                     off_fwd, read_ids=batch["read_ids"], cand_ids=cand_ids)
            rc = self.banded_scorer(rc_emb, batch["read_rc_mask"], c_emb, c_sal, cand_mask,
                                    off_rc, read_ids=batch["read_rc_ids"], cand_ids=cand_ids)
            out[g] = (torch.maximum(fwd, rc), sub_target)
        return out

    # ---- inference ----
    @torch.no_grad()
    def encode_reference(self, ref, seed_k=9):
        """Encode a reference set once (cache candidate embeddings + groups + k-mer seed index)."""
        dev = self.device
        present = [g for g in GENES if g in ref.genes]
        groups = {g: build_sequence_groups(ref.gene(g)) for g in present}
        cand = {}
        cand_ids = {}
        for g in present:
            cids, cmask = encode_batch(groups[g].unique_seqs, self.cfg.max_len)
            cemb, csal = self.encoder(cids.to(dev), cmask.to(dev))
            cand[g] = (cemb, csal, cmask.to(dev))
            cand_ids[g] = cids.to(dev)
        seed = {g: KmerIndex(groups[g].unique_seqs, k=seed_k) for g in present}
        return {"present": present, "groups": groups, "cand": cand, "seed": seed,
                "cand_ids": cand_ids}

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
        cand_ids = cached.get("cand_ids", {})
        rc = tk.revcomp(seq)
        read = seq if seed["V"].best_hits(seq) >= seed["V"].best_hits(rc) else rc
        shortlists = {g: (seed[g].shortlist(read, topk)[0] if g == "V" else list(range(groups[g].n)))
                      for g in present}

        def gene_fine(g):
            if fine == "align" or (fine == "hybrid" and g == "V"):
                return "align"
            return "neural"

        r_emb = r_mask = r_ids = None
        if any(gene_fine(g) == "neural" for g in present):     # encode the read only if any gene needs it
            dev = self.device
            r_ids, rm = encode_batch([read], self.cfg.max_len)
            r_emb, _ = self.encoder(r_ids.to(dev), rm.to(dev))
            r_mask = rm.to(dev)
            r_ids = r_ids.to(dev)

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
                if self.banded_scorer is None:
                    scores = masked_maxsim(r_emb[0], r_mask[0], cemb[ix], csal[ix], cmask[ix])
                else:
                    offsets = torch.tensor([seed[g].anchor_offsets(read, idxs)],
                                           dtype=torch.long, device=self.device)
                    scores = self.banded_scorer(
                        r_emb, r_mask, cemb[ix], csal[ix], cmask[ix], offsets,
                        read_ids=r_ids, cand_ids=cand_ids[g][ix])[0]
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
        if self.banded_scorer is None:
            vfwd = masked_maxsim(r_emb[0], r_mask[0], *cand["V"])
            vrc = masked_maxsim(rc_emb[0], rc_mask[0], *cand["V"])
        else:
            cand_ids = cached["cand_ids"]
            v_idxs = list(range(groups["V"].n))
            voff_fwd = torch.tensor([cached["seed"]["V"].anchor_offsets(seq, v_idxs)],
                                    dtype=torch.long, device=dev)
            voff_rc = torch.tensor([cached["seed"]["V"].anchor_offsets(tk.revcomp(seq), v_idxs)],
                                   dtype=torch.long, device=dev)
            vfwd = self.banded_scorer(
                r_emb, r_mask, *cand["V"], voff_fwd, read_ids=r_ids.to(dev),
                cand_ids=cand_ids["V"])[0]
            vrc = self.banded_scorer(
                rc_emb, rc_mask, *cand["V"], voff_rc, read_ids=rc_ids.to(dev),
                cand_ids=cand_ids["V"])[0]
        strand = pick_strand(vfwd, vrc)
        emb, mask = (r_emb[0], r_mask[0]) if strand == 0 else (rc_emb[0], rc_mask[0])
        batch_emb, batch_mask = (r_emb, r_mask) if strand == 0 else (rc_emb, rc_mask)
        read_seq = seq if strand == 0 else tk.revcomp(seq)
        read_ids = r_ids.to(dev) if strand == 0 else rc_ids.to(dev)
        out = {}
        for g in present:
            if self.banded_scorer is None:
                scores = masked_maxsim(emb, mask, *cand[g])
            else:
                idxs = list(range(groups[g].n))
                offsets = torch.tensor([cached["seed"][g].anchor_offsets(read_seq, idxs)],
                                       dtype=torch.long, device=dev)
                scores = self.banded_scorer(
                    batch_emb, batch_mask, *cand[g], offsets,
                    read_ids=read_ids, cand_ids=cached["cand_ids"][g])[0]
            gid = int(scores.argmax())
            out[g] = {"names": infer.called_names(gid, groups[g]),
                      "seq": groups[g].unique_seqs[gid]}
        return out

    @torch.no_grad()
    def predict(self, ref, seq: str):
        return self.predict_cached(self.encode_reference(ref), seq)

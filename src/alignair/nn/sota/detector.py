"""Open-vocabulary VDJ detector — the assembly.

Composes the SOTA building blocks into one model that maps
    a read + a candidate germline set (the open vocabulary, given at inference)
  -> per gene: {allele call, in-read span, objectness, germline trims}.
Everything else in the GenAIRR record is derived downstream (deterministic post-process).

Pipeline (YOLO-World / GLIP shape, 1-D DNA):
  1. SharedNucleotideEncoder encodes the read and every candidate germline allele.
  2. ReferenceFusion (GLIP) fuses the read against the candidate vocabulary, so the read
     representation is conditioned on THIS genotype — novel / renamed alleles included.
  3. TypedVDJDecoder (DETR) decodes three fixed typed queries (V, D, J) from the fused read.
  4. Decoupled heads (YOLOX split): the query vector drives *localization* (SpanHead span +
     objectness, trim regression); *identity* is a token-level TokenMatch (CLIP/ColBERT) of the
     fused read tokens against each gene's candidate tokens — the allele call.

Why this split beats pooling for siblings: fusion (learned, + hard negatives upstream) conditions
the read on the genotype, and the MaxSim match keeps per-token detail so the 1-2 SNPs separating
sibling alleles survive instead of being averaged away. The genotype is dynamic: the candidate
set IS the reference the caller passes, so novel / renamed alleles work with no retrain.

Contract: candidates are a bounded set per gene (post-retrieval top-k, or the full small set for
D/J), shared across the batch (one genotype). Candidate token reps are L2-normalized before MaxSim.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import ReferenceFusion
from .query_decoder import TypedVDJDecoder, GENES
from .span_head import SpanHead
from .matching import TokenMatch
from .retrieval import retrieve_topk, gather_candidates
from ..encoder.shared import SharedNucleotideEncoder


class OpenVocabVDJDetector(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, encoder_layers: int = 6,
                 fusion_layers: int = 2, decoder_layers: int = 3, vocab_size: int = 6,
                 max_len: int = 1024):
        super().__init__()
        self.encoder = SharedNucleotideEncoder(d_model, encoder_layers, nhead,
                                               vocab_size=vocab_size, max_len=max_len)
        # read-only conditioning: we consume the fused read, not the fused candidates
        # (identity comes from MaxSim over raw candidate tokens), so fuse unidirectionally.
        self.fusion = ReferenceFusion(d_model, nhead, n_layers=fusion_layers, bidirectional=False)
        self.decoder = TypedVDJDecoder(d_model, nhead, n_layers=decoder_layers)
        self.span_heads = nn.ModuleDict({g: SpanHead(d_model) for g in GENES})
        self.trim_heads = nn.ModuleDict({
            g: nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 2))
            for g in GENES})                                   # (5' trim, 3' trim), normalized
        self.match = nn.ModuleDict({g: TokenMatch() for g in GENES})
        # retriever temperature (CLIP logit_scale): trains the pooled read<->candidate similarity
        # that drives the top-k prefilter, so the true allele is actually surfaced at inference.
        self.retr_log_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def _encode_candidates(self, tokens, mask):
        tok = self.encoder.forward_positions(tokens, mask, SharedNucleotideEncoder.GERMLINE)
        m = mask.unsqueeze(-1).to(tok.dtype)
        pooled = (tok * m).sum(1) / m.sum(1).clamp(min=1.0)    # (K, d) — for fusion conditioning
        return tok, pooled                                     # tokens (K, Sc, d) — for MaxSim

    def forward(self, read_tokens, read_mask, candidates: dict, top_k: int | None = None) -> dict:
        """read_tokens (B, L), read_mask (B, L) True=valid.
        candidates[gene] = {'tokens': (Kg, Sc), 'mask': (Kg, Sc) True=valid,
                            'candidate_mask': (Kg,) bool optional — genotype restriction,
                            'force_include': (B,) long optional — training positive to keep in top-k}.
        top_k: if set and a gene's reference exceeds it, retrieve the top-k shortlist (cheap pooled
               cosine) and discriminate only those; allele_scores are scattered back to full (B, Kg)
               with non-retrieved alleles at -inf, so multi-hot targets over the full reference work.
        -> {gene: {'span','objectness','allele_scores','trim'}}."""
        B = read_tokens.shape[0]
        read_tok = self.encoder.forward_positions(read_tokens, read_mask,
                                                  SharedNucleotideEncoder.READ)      # (B, L, d)
        rm = read_mask.unsqueeze(-1).to(read_tok.dtype)
        read_pooled = F.normalize((read_tok * rm).sum(1) / rm.sum(1).clamp(min=1.0), dim=-1)

        # per gene: (shortlist candidate tokens/mask, pooled shortlist for fusion, index map or None)
        shortlist, vocab_parts, retr_full = {}, [], {}
        for g in GENES:
            ctok, cpool = self._encode_candidates(candidates[g]["tokens"], candidates[g]["mask"])
            Kg = ctok.shape[0]
            cpool_n = F.normalize(cpool, dim=-1)
            # full-reference retrieval logits (pooled cosine × temp) — trains the prefilter.
            retr_full[g] = (read_pooled @ cpool_n.t()) * self.retr_log_scale.exp().clamp(max=100.0)
            if top_k is not None and Kg > top_k:
                idx = retrieve_topk(read_pooled, cpool_n, top_k,
                                    candidates[g].get("force_include"))              # (B, k)
                gtok, gmsk = gather_candidates(ctok, candidates[g]["mask"], idx)     # (B,k,Sc,d),(B,k,Sc)
                gpool = cpool[idx]                                                   # (B, k, d)
            else:
                idx = None
                gtok = ctok.unsqueeze(0).expand(B, -1, -1, -1)
                gmsk = candidates[g]["mask"].unsqueeze(0).expand(B, -1, -1)
                gpool = cpool.unsqueeze(0).expand(B, -1, -1)                         # (B, Kg, d)
            shortlist[g] = (gtok, gmsk, idx, Kg)
            vocab_parts.append(gpool)

        vocab = torch.cat(vocab_parts, dim=1)                                        # (B, sum k, d)
        fused_read, _ = self.fusion(read_tok, vocab, read_mask, None)                # genotype-cond.
        queries = self.decoder(fused_read, read_mask)                               # {gene: (B, d)}
        fr = F.normalize(fused_read, dim=-1)

        out = {}
        for g in GENES:
            gtok, gmsk, idx, Kg = shortlist[g]
            sh = self.span_heads[g](queries[g])
            scores = self.match[g].score_batched(fr, read_mask, F.normalize(gtok, dim=-1), gmsk)
            if idx is not None:                                                      # scatter to full ref
                full = scores.new_full((B, Kg), float("-inf"))
                scores = full.scatter(1, idx, scores)
            retr = retr_full[g]
            cmask = candidates[g].get("candidate_mask")
            if cmask is not None:
                m = ~cmask[None].to(scores.device)
                scores = scores.masked_fill(m, float("-inf"))
                retr = retr.masked_fill(m, float("-inf"))
            out[g] = {"span": sh["span"], "objectness": sh["objectness"],
                      "allele_scores": scores, "retrieval_scores": retr,
                      "trim": torch.sigmoid(self.trim_heads[g](queries[g]))}
        return out

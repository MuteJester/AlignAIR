"""Raw-read inference for DNAlignAIR.

Takes plain read strings and returns per-read predictions (V/D/J calls + in-sequence
and germline start/end) in GenAIRR ground-truth coordinate convention (0-based start,
position-style end), so they can be scored by the IgBLAST harness directly. This is
the DEPLOYED path: end-to-end (predicted orientation, predicted regions, predicted
top-1 allele), no teacher forcing.
"""
import torch

from ..data.tokenizer import pad_tokenize
from ..nn.region_head import decode_boundaries
from ..nn.germline_aligner import decode_germline_coords
from ..training.germline_tf import compute_germline_logits
from ..core.dnalignair import extract_segment_tokens


_ALIGNER = None


def _aligner():
    global _ALIGNER
    if _ALIGNER is None:
        from Bio.Align import PairwiseAligner
        a = PairwiseAligner(mode="local")          # Smith-Waterman, indel-aware
        a.match_score, a.mismatch_score = 1.0, -1.0
        a.open_gap_score, a.extend_gap_score = -2.0, -0.5
        _ALIGNER = a
    return _ALIGNER


def rescore_alleles(reads, preds, reference_set, genes=("v", "d")) -> list:
    """Resolve the exact allele within the model's predicted gene by GAPPED local
    alignment (mini-IgBLAST): the neural model gets the GENE and coordinates right;
    re-rank the gene's sibling alleles by Smith-Waterman alignment score of the observed
    segment to each candidate germline. Unlike a rigid position compare, this is
    indel-robust (the failure mode under SHM). Pure post-processing; mutates preds."""
    aligner = _aligner()
    for read, p in zip(reads, preds):
        read = str(read).upper()
        for g in genes:
            G = g.upper()
            top1 = p.get(f"{g}_call")
            if not top1:
                continue
            ref = reference_set.gene(G)
            topk = p.get(f"{g}_topk")
            if topk:  # rerank the top-k embedding candidates ACROSS genes (fixes gene errors)
                cand_names = topk
            else:     # fall back to siblings of the predicted gene
                gene = top1.split("*")[0]
                cand_names = [nm for nm in ref.names if nm.split("*")[0] == gene]
            cands = [(nm, ref.sequences[ref.index[nm]]) for nm in cand_names]
            if len(cands) <= 1:
                continue
            ss, se = p[f"{g}_sequence_start"], p[f"{g}_sequence_end"]
            obs = read[ss:se]
            if len(obs) < 5:
                continue
            best, best_s = top1, float("-inf")
            for nm, germ in cands:
                s = aligner.score(obs, germ)
                if s > best_s:
                    best_s, best = s, nm
            p[f"{g}_call"] = best
    return preds


@torch.no_grad()
def predict_reads(model, reference_set, reads, device=None, batch_size: int = 64,
                  topk: int = 16, rerank: str = "none") -> list:
    """rerank: 'none' (stage-1 top-1), or 'learned' (rerank top-k by the in-model
    differentiable aligner.alignment_score = the learned allele reader)."""
    device = device or next(model.parameters()).device
    model.eval()
    ref_emb = model.encode_reference(reference_set)
    has_d = reference_set.has_d
    genes = ["v", "j"] + (["d"] if has_d else [])
    names = {g.upper(): reference_set.gene(g.upper()).names for g in genes}

    preds = []
    for s in range(0, len(reads), batch_size):
        chunk = reads[s:s + batch_size]
        tokens, mask = pad_tokenize(chunk)
        tokens, mask = tokens.to(device), mask.to(device)
        out = model(tokens, mask, ref_emb)                  # end-to-end (predicted orientation)
        canon = out["canon_tokens"]
        boundary = out.get("boundary")
        dec = decode_boundaries(out["region_logits"], mask, has_d=has_d)
        pred_region = out["region_logits"].argmax(-1)
        pred_idx = {g.upper(): out["match"][g.upper()].argmax(-1) for g in genes}
        topk_idx = {g.upper(): out["match"][g.upper()].topk(
            min(topk, len(names[g.upper()])), dim=-1).indices for g in genes}
        gl = compute_germline_logits(model, canon, mask, {}, ref_emb, has_d,
                                     region_labels=pred_region, allele_idx=pred_idx)
        gcoord = {g: decode_germline_coords(gl[g][0], gl[g][1]) for g in genes}
        # learned allele reader: rerank top-k by the differentiable alignment score
        learned_best = {}
        if rerank == "learned":
            for g in genes:
                G = g.upper()
                seg_tok, seg_mask = extract_segment_tokens(canon, mask, pred_region, G)
                seg_pos = model.germline_encoder.forward_positions(seg_tok, seg_mask)  # (B,S,d)
                pos_reps, pos_mask = ref_emb[G]["pos_reps"], ref_emb[G]["pos_mask"]
                chosen = []
                for i in range(len(chunk)):
                    cands = topk_idx[G][i]                              # (k,)
                    k = cands.shape[0]
                    sc = model.aligner.alignment_score(
                        seg_pos[i:i + 1].expand(k, -1, -1), seg_mask[i:i + 1].expand(k, -1),
                        pos_reps[cands], pos_mask[cands])               # (k,)
                    chosen.append(int(cands[sc.argmax()]))
                learned_best[G] = chosen
        for i in range(len(chunk)):
            p = {}
            for g in genes:
                G = g.upper()
                idx = learned_best[G][i] if rerank == "learned" else int(pred_idx[G][i])
                p[f"{g}_call"] = names[G][idx]
                p[f"{g}_topk"] = [names[G][int(j)] for j in topk_idx[G][i]]
                if boundary is not None:                    # query decoder: posterior argmax
                    p[f"{g}_sequence_start"] = int(boundary["start"][G][i].argmax())
                    p[f"{g}_sequence_end"] = int(boundary["end"][G][i].argmax()) + 1
                else:
                    p[f"{g}_sequence_start"] = dec[i][f"{g}_start"]
                    p[f"{g}_sequence_end"] = dec[i][f"{g}_end"]
                p[f"{g}_germline_start"] = int(gcoord[g][0][i])
                p[f"{g}_germline_end"] = int(gcoord[g][1][i])
            preds.append(p)
    return preds

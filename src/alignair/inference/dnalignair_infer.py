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
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def canonicalize_sequence(seq: str, orientation_id: int) -> str:
    """Apply the model's predicted orientation transform to recover the FORWARD/canonical
    sequence (the frame predict_reads coordinates are in). Mirrors nn.orientation transform
    ids: 0=identity, 1=reverse-complement, 2=complement, 3=reverse (all involutions)."""
    s = seq.upper()
    if orientation_id == 1:        # REVERSE_COMPLEMENT (complement then reverse)
        return s.translate(_COMPLEMENT)[::-1]
    if orientation_id == 2:        # COMPLEMENT
        return s.translate(_COMPLEMENT)
    if orientation_id == 3:        # REVERSE
        return s[::-1]
    return s                       # IDENTITY


def resolve_hierarchy(call_set, top1, max_allele: int = 3):
    """Graceful degradation over the calibrated equivalence set: report the MOST SPECIFIC
    level the evidence supports — allele if the set is small, else the shared gene, else
    the shared family, else abstain. Returns (resolved_call, level). Levels:
    'allele' > 'gene' > 'family' > 'none' (abstain). Short fragments carry little V, so the
    set spans many alleles; this turns a near-random allele guess into a correct coarser call."""
    s = [c for c in (call_set or []) if c]
    if not s:
        return (top1, "allele") if top1 else (None, "none")
    if len(s) <= max_allele and len(set(s)) <= max_allele:
        genes = {c.split("*")[0] for c in s}
        if len(s) == 1:
            return s[0], "allele"
        if len(genes) == 1:                       # small set, one gene -> gene-level
            return genes.pop(), "gene"
    genes = {c.split("*")[0] for c in s}
    if len(genes) == 1:
        return genes.pop(), "gene"
    families = {c.split("-")[0] for c in s}
    if len(families) == 1:
        return families.pop(), "family"
    return None, "none"                            # spans families -> abstain


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
                  topk: int = 16, rerank: str = "none", set_epsilon: float = 1.0,
                  genotype: dict | None = None, calibration: dict | None = None,
                  emit_scores: bool = False, state_conditioning: bool = True,
                  rerank_chunk: int = 2048, contaminant_tau: float | None = None) -> list:
    """rerank: 'none' (stage-1 top-1), or 'learned' (rerank top-k by the in-model
    differentiable aligner.alignment_score = the learned allele reader). When rerank
    is on, also emits {g}_call_set = the calibrated equivalence set (candidates within
    set_epsilon of the top score) — the multi-label output (report the set, not argmax,
    when the evidence cannot distinguish alleles).

    The equivalence set is a temperature-scaled log-likelihood-ratio band: keep candidate c
    iff (s_top - s_c)/T <= epsilon. `calibration` = {GENE: {temperature, epsilon}} (from
    benchmark.evaluation.allele_calibration) overrides T=1 / epsilon=set_epsilon per gene;
    the per-read score offset from the state-conditioned emission cancels in s_top - s_c, so
    the band is invariant to it. emit_scores adds {g}_scores=[(name, raw_score)] for calibration.

    genotype: optional DYNAMIC reference restriction {gene_type: [allowed allele names]}
    with gene_type in {'v','d','j'}. Alleles outside the genotype are scored -inf and can
    never be called — so a donor's allele subset conditions every prediction. (For NOVEL
    alleles, build the reference_set itself from the genotype via ReferenceSet.from_yaml;
    then pass genotype=None or the same names.) Genes absent from `genotype` stay full."""
    device = device or next(model.parameters()).device
    model.eval()
    ref_emb = model.encode_reference(reference_set)
    has_d = reference_set.has_d
    genes = ["v", "j"] + (["d"] if has_d else [])
    names = {g.upper(): reference_set.gene(g.upper()).names for g in genes}

    # dynamic genotype mask (static across reads in this call): -inf any allele not allowed,
    # and cap top-k to the allowed count so the learned reranker never sees a disallowed one.
    candidate_masks = None
    n_allowed = {g.upper(): len(names[g.upper()]) for g in genes}
    if genotype is not None:
        gt = {k.upper(): set(v) for k, v in genotype.items()}
        candidate_masks = {}
        for g in genes:
            G = g.upper()
            if G not in gt:
                continue
            m = reference_set.genotype_mask(G, gt[G])
            if int(m.sum()) == 0:
                raise ValueError(f"genotype for gene {G} excludes every allele in the reference")
            candidate_masks[G] = m.to(device)
            n_allowed[G] = int(m.sum())

    # out-of-scope / contaminant gate (flag-only): a read whose best length-normalized V
    # alignment quality falls below tau is flagged is_contaminant=True (calls are RETAINED,
    # never deleted). tau is a calibrated threshold (calibration['contaminant']['tau']).
    contam_tau = (contaminant_tau if contaminant_tau is not None
                  else (calibration or {}).get("contaminant", {}).get("tau"))

    preds = []
    for s in range(0, len(reads), batch_size):
        chunk = reads[s:s + batch_size]
        tokens, mask = pad_tokenize(chunk)
        tokens, mask = tokens.to(device), mask.to(device)
        out = model(tokens, mask, ref_emb, candidate_masks=candidate_masks)  # end-to-end
        canon = out["canon_tokens"]
        boundary = out.get("boundary")
        dec = decode_boundaries(out["region_logits"], mask, has_d=has_d)
        pred_region = out["region_logits"].argmax(-1)
        pred_idx = {g.upper(): out["match"][g.upper()].argmax(-1) for g in genes}
        topk_idx = {g.upper(): out["match"][g.upper()].topk(
            min(topk, n_allowed[g.upper()]), dim=-1).indices for g in genes}
        gl = compute_germline_logits(model, canon, mask, {}, ref_emb, has_d,
                                     region_labels=pred_region, allele_idx=pred_idx)
        gcoord = {g: decode_germline_coords(gl[g][0], gl[g][1]) for g in genes}
        # learned allele reader: rerank top-k by the differentiable alignment score
        learned_best = {}
        if rerank == "learned":
            from ..nn.state_head import state_reliability
            from ..core.dnalignair import extract_segment
            cal = calibration or {}
            for g in genes:
                G = g.upper()
                T = float(cal.get(G, {}).get("temperature", 1.0))
                eps = float(cal.get(G, {}).get("epsilon", set_epsilon))
                seg_tok, seg_mask = extract_segment_tokens(canon, mask, pred_region, G)
                seg_pos = model.germline_encoder.forward_positions(seg_tok, seg_mask)  # (B,S,d)
                if state_conditioning:
                    seg_state, _ = extract_segment(out["state_logits"], mask, pred_region, G)
                    seg_rel = state_reliability(seg_state)              # (B,S) SHM down-weight
                else:
                    seg_rel = None
                pos_reps, pos_mask = ref_emb[G]["pos_reps"], ref_emb[G]["pos_mask"]
                pos_tok = ref_emb[G]["pos_tok"]
                # VECTORIZED rerank: score every (read, candidate) pair in batched soft-DP
                # calls instead of a per-read Python loop. Each alignment is independent, so
                # this is the same per-item math, just GPU-parallel (chunked to bound memory).
                B = len(chunk)
                ti = topk_idx[G]                                       # (B, k)
                kk = ti.shape[1]
                read_ix = torch.arange(B, device=device).repeat_interleave(kk)   # (B*k,)
                cand_ix = ti.reshape(-1)                               # (B*k,)
                parts = []
                for a in range(0, B * kk, rerank_chunk):
                    sl = slice(a, a + rerank_chunk)
                    ri, ci = read_ix[sl], cand_ix[sl]
                    parts.append(model.aligner.alignment_score(
                        seg_pos[ri], seg_mask[ri], pos_reps[ci], pos_mask[ci],
                        seg_tok=seg_tok[ri], germ_tok=pos_tok[ci],
                        seg_reliability=(seg_rel[ri] if seg_rel is not None else None)))
                sc_all = torch.cat(parts).reshape(B, kk)              # (B, k) candidate scores
                # temperature-scaled log-likelihood-ratio band (codex): keep iff the top is at
                # most exp(eps) times likelier; calibrated per gene, T=1 default.
                delta = (sc_all.max(dim=1, keepdim=True).values - sc_all) / T   # (B,k)
                keep = delta <= eps                                   # (B,k)
                conf = (torch.softmax(sc_all / T, dim=1) * keep).sum(dim=1)     # (B,) mass in set
                ti_l, keep_l = ti.cpu().tolist(), keep.cpu().tolist()
                top_l = sc_all.argmax(dim=1).cpu().tolist()
                learned_best[G] = [ti_l[i][top_l[i]] for i in range(B)]
                learned_best[G + "_set"] = [
                    [names[G][ti_l[i][j]] for j in range(kk) if keep_l[i][j]] for i in range(B)]
                learned_best[G + "_conf"] = conf.cpu().tolist()
                if G == "V":  # out-of-scope gate: best LENGTH-NORMALIZED V alignment quality
                    seg_len = seg_mask.sum(dim=1).clamp(min=1).to(sc_all.dtype)  # (B,) predicted V len
                    learned_best["_gate"] = (sc_all.max(dim=1).values / seg_len).cpu().tolist()
                if emit_scores:
                    sc_l = sc_all.cpu().tolist()
                    learned_best[G + "_scores"] = [
                        [(names[G][ti_l[i][j]], sc_l[i][j]) for j in range(kk)] for i in range(B)]
        for i in range(len(chunk)):
            p = {}
            for g in genes:
                G = g.upper()
                idx = learned_best[G][i] if rerank == "learned" else int(pred_idx[G][i])
                p[f"{g}_call"] = names[G][idx]
                p[f"{g}_topk"] = [names[G][int(j)] for j in topk_idx[G][i]]
                if rerank == "learned":
                    cset = learned_best[G + "_set"][i]
                    p[f"{g}_call_set"] = cset
                    p[f"{g}_calls"] = cset                             # benchmark-adapter alias
                    p[f"{g}_set_confidence"] = learned_best[G + "_conf"][i]
                    # graceful hierarchical degradation + abstention from the calibrated set
                    resolved, level = resolve_hierarchy(cset, p[f"{g}_call"])
                    p[f"{g}_resolved_call"] = resolved
                    p[f"{g}_call_level"] = level
                    if emit_scores:
                        p[f"{g}_scores"] = learned_best[G + "_scores"][i]
                if boundary is not None:                    # query decoder: posterior argmax
                    p[f"{g}_sequence_start"] = int(boundary["start"][G][i].argmax())
                    p[f"{g}_sequence_end"] = int(boundary["end"][G][i].argmax()) + 1
                else:
                    p[f"{g}_sequence_start"] = dec[i][f"{g}_start"]
                    p[f"{g}_sequence_end"] = dec[i][f"{g}_end"]
                p[f"{g}_germline_start"] = int(gcoord[g][0][i])
                p[f"{g}_germline_end"] = int(gcoord[g][1][i])
            # sequence-level annotations (predicted orientation + scalar heads)
            p["orientation_id"] = int(out["orientation_logits"][i].argmax())
            p["productive"] = bool(out["productive"][i].item() > 0.5)
            p["mutation_rate"] = float(out["mutation_rate"][i].item())
            p["indel_count"] = float(out["indel_count"][i].item())
            # out-of-scope flag (advisory; calls above are RETAINED regardless)
            if rerank == "learned" and "_gate" in learned_best:
                gs = learned_best["_gate"][i]
                p["contaminant_score"] = float(gs)
                if contam_tau is not None:
                    p["is_contaminant"] = bool(gs < contam_tau)
            preds.append(p)
    return preds

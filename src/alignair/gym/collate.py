"""Collate gym target bundles into batched tensors."""
import torch

IGNORE = -100  # CrossEntropyLoss ignore_index for padded per-position labels


def _multihot(call_sets, gene_ref):
    n = len(gene_ref.names)
    out = torch.zeros(len(call_sets), n, dtype=torch.float32)
    for i, names in enumerate(call_sets):
        for nm in names:
            j = gene_ref.index.get(nm)
            if j is not None:
                out[i, j] = 1.0
    return out


def gym_collate(batch, reference_set, has_d: bool):
    B = len(batch)
    lmax = max(len(s["tokens"]) for s in batch)

    tokens = torch.zeros(B, lmax, dtype=torch.long)
    mask = torch.zeros(B, lmax, dtype=torch.bool)
    region = torch.full((B, lmax), IGNORE, dtype=torch.long)
    state = torch.full((B, lmax), IGNORE, dtype=torch.long)
    for i, s in enumerate(batch):
        n = len(s["tokens"])
        tokens[i, :n] = torch.from_numpy(s["tokens"])
        mask[i, :n] = True
        region[i, :n] = torch.from_numpy(s["region_labels"])
        state[i, :n] = torch.from_numpy(s["state_labels"])

    genes = ["v", "j"] + (["d"] if has_d else [])
    out = {"tokens": tokens, "mask": mask, "region_labels": region, "state_labels": state}
    # A sample may legitimately lack a gene (a light-chain read in a mixed-chain batch has no
    # D). Such samples get sentinel coords / empty calls and are excluded from that gene's
    # supervision (below) rather than crashing the collate.
    for g in genes:
        G = g.upper()
        gref = reference_set.gene(G)
        pres = [g in s["germline"] for s in batch]
        out[f"{g}_germline_start"] = torch.tensor(
            [s["germline"][g][0] if p else 0 for s, p in zip(batch, pres)], dtype=torch.long)
        out[f"{g}_germline_end"] = torch.tensor(
            [s["germline"][g][1] if p else 0 for s, p in zip(batch, pres)], dtype=torch.long)
        out[f"{g}_start"] = torch.tensor(
            [s["inseq"][g][0] if p else 0 for s, p in zip(batch, pres)], dtype=torch.long)
        out[f"{g}_end"] = torch.tensor(
            [s["inseq"][g][1] if p else 0 for s, p in zip(batch, pres)], dtype=torch.long)
        out[f"{g}_allele"] = _multihot(
            [s["calls"].get(G, []) if p else [] for s, p in zip(batch, pres)], gref)

        def _primary(s, p):
            if not p:
                return 0
            nm = s.get("primary", {}).get(G)
            if nm is None:  # fall back to any listed call (synthetic/legacy bundles)
                names = s["calls"].get(G)
                nm = next(iter(names)) if names else None
            return gref.index.get(nm, 0)

        out[f"{g}_primary_idx"] = torch.tensor(
            [_primary(s, p) for s, p in zip(batch, pres)], dtype=torch.long)

    if has_d:
        # exclude from D supervision: inverted-D rows (RC vs forward reference) AND samples
        # that have no D at all (e.g. light chains). Zeroing the multi-hot makes the
        # contrastive D-match contribute 0 for them.
        supervise = torch.tensor(
            [0.0 if (s.get("d_inverted") or "d" not in s["germline"]) else 1.0 for s in batch])
        out["d_supervise"] = supervise
        out["d_allele"] = out["d_allele"] * supervise.unsqueeze(1)

    out["orientation_id"] = torch.tensor([s["orientation_id"] for s in batch], dtype=torch.long)

    # teacher (full forward read) view for EMA self-distillation, padded separately
    if "teacher_tokens" in batch[0]:
        tl = max(len(s["teacher_tokens"]) for s in batch)
        t_tok = torch.zeros(B, tl, dtype=torch.long)
        t_mask = torch.zeros(B, tl, dtype=torch.bool)
        for i, s in enumerate(batch):
            n = len(s["teacher_tokens"])
            t_tok[i, :n] = torch.from_numpy(s["teacher_tokens"])
            t_mask[i, :n] = True
        out["teacher_tokens"] = t_tok
        out["teacher_mask"] = t_mask

    for key in ("noise_count", "mutation_rate", "indel_count", "productive"):
        out[key] = torch.tensor([[s[key]] for s in batch], dtype=torch.float32)
    return out

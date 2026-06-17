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
    for g in genes:
        out[f"{g}_germline_start"] = torch.tensor([s["germline"][g][0] for s in batch], dtype=torch.long)
        out[f"{g}_germline_end"] = torch.tensor([s["germline"][g][1] for s in batch], dtype=torch.long)
        out[f"{g}_start"] = torch.tensor([s["inseq"][g][0] for s in batch], dtype=torch.long)
        out[f"{g}_end"] = torch.tensor([s["inseq"][g][1] for s in batch], dtype=torch.long)
        out[f"{g}_allele"] = _multihot([s["calls"][g.upper()] for s in batch],
                                       reference_set.gene(g.upper()))
        gref = reference_set.gene(g.upper())

        def _primary(s):
            nm = s.get("primary", {}).get(g.upper())
            if nm is None:  # fall back to any listed call (synthetic/legacy bundles)
                names = s["calls"].get(g.upper())
                nm = next(iter(names)) if names else None
            return gref.index.get(nm, 0)

        out[f"{g}_primary_idx"] = torch.tensor([_primary(s) for s in batch], dtype=torch.long)

    if has_d:
        # mask inverted-D rows out of D supervision (RC vs forward reference);
        # zeroing the multi-hot makes the contrastive D-match contribute 0 for them.
        supervise = torch.tensor([0.0 if s.get("d_inverted") else 1.0 for s in batch])
        out["d_supervise"] = supervise
        out["d_allele"] = out["d_allele"] * supervise.unsqueeze(1)

    out["orientation_id"] = torch.tensor([s["orientation_id"] for s in batch], dtype=torch.long)
    for key in ("noise_count", "mutation_rate", "indel_count", "productive"):
        out[key] = torch.tensor([[s[key]] for s in batch], dtype=torch.float32)
    return out

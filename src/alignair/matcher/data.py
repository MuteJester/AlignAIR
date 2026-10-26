"""Data bridge: sequence-group collapse, set-aware targets, batching, gym streaming."""
from dataclasses import dataclass
from typing import Dict, Iterator, List

import torch

from .config import MatcherConfig
from .seed import KmerIndex
from . import tokenizer as tk

GENES = ("V", "D", "J")


@dataclass
class SequenceGroups:
    unique_seqs: List[str]
    group_names: List[List[str]]
    name_to_group: Dict[str, int]

    @property
    def n(self) -> int:
        return len(self.unique_seqs)


def build_sequence_groups(gene_ref) -> SequenceGroups:
    seq_to_gid: Dict[str, int] = {}
    unique, group_names, name_to_group = [], [], {}
    for name, seq in zip(gene_ref.names, gene_ref.sequences):
        gid = seq_to_gid.get(seq)
        if gid is None:
            gid = len(unique)
            seq_to_gid[seq] = gid
            unique.append(seq)
            group_names.append([])
        name_to_group[name] = gid
        group_names[gid].append(name)
    return SequenceGroups(unique, group_names, name_to_group)


def target_mask(truth_names: List[str], groups: SequenceGroups) -> torch.Tensor:
    m = torch.zeros(groups.n, dtype=torch.bool)
    for nm in truth_names:
        gid = groups.name_to_group.get(nm)
        if gid is not None:
            m[gid] = True
    return m


def encode_batch(seqs: List[str], max_len: int):
    L = min(max(len(s) for s in seqs), max_len)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    mask = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        e = tk.encode(s)[:L]
        ids[i, : e.numel()] = e
        mask[i, : e.numel()] = True
    return ids, mask


def _split_calls(value) -> List[str]:
    return [x for x in str(value or "").split(",") if x]


def stream_batches(dataconfig, cfg: MatcherConfig, params: dict, batch_size: int,
                   seed: int, rc_prob: float = 0.5) -> Iterator[dict]:
    """Yield config-homogeneous training batches from a single dataconfig.

    Reference R is built once from this dataconfig; inverted-D is disabled; reads are
    forward/revcomp-augmented. Emitted batch dict schema:
        read_ids [B,Ls], read_mask [B,Ls], read_rc_ids [B,Ls], read_rc_mask [B,Ls],
        genes: {G: {cand_ids [C,Lc], cand_mask [C,Lc], target [B,C] bool, groups}}
    """
    import random

    from alignair.reference.reference_set import ReferenceSet
    from alignair.train.gym.experiment import build_experiment

    ref = ReferenceSet.from_dataconfigs(dataconfig)
    present = [g for g in GENES if g in ref.genes]
    groups = {g: build_sequence_groups(ref.gene(g)) for g in present}
    cand = {g: encode_batch(groups[g].unique_seqs, cfg.max_len) for g in present}
    # banded scorer needs per-(read,candidate) seed offsets; skip the cost for maxsim.
    seed_idx = ({g: KmerIndex(groups[g].unique_seqs, k=9) for g in present}
                if cfg.scorer == "banded" else None)

    p = dict(params)
    p["invert_d_prob"] = 0.0                       # disable inverted-D in v1
    exp = build_experiment(dataconfig, p, allow_curatable=True)
    rng = random.Random(seed)

    buf: List[dict] = []
    for rec in exp.stream_records(n=None, seed=seed):
        seq = str(rec["sequence"]).upper()
        truth = {g: _split_calls(rec.get(f"{g.lower()}_call")) for g in present}
        if rc_prob and rng.random() < rc_prob:
            seq = tk.revcomp(seq)
        buf.append({"seq": seq, "truth": truth})
        if len(buf) < batch_size:
            continue
        read_ids, read_mask = encode_batch([b["seq"] for b in buf], cfg.max_len)
        read_rc_ids, read_rc_mask = encode_batch([tk.revcomp(b["seq"]) for b in buf], cfg.max_len)
        genes = {}
        for g in present:
            tgt = torch.stack([target_mask(b["truth"][g], groups[g]) for b in buf])  # [B, n_g]
            genes[g] = {"cand_ids": cand[g][0], "cand_mask": cand[g][1], "target": tgt,
                        "groups": groups[g]}
            if seed_idx is not None:                # [B, C] anchor offsets (read_start - cand_start)
                allc = list(range(groups[g].n))
                genes[g]["offsets_fwd"] = torch.tensor(
                    [seed_idx[g].anchor_offsets(b["seq"], allc) for b in buf], dtype=torch.long)
                genes[g]["offsets_rc"] = torch.tensor(
                    [seed_idx[g].anchor_offsets(tk.revcomp(b["seq"]), allc) for b in buf], dtype=torch.long)
        yield {"read_ids": read_ids, "read_mask": read_mask,
               "read_rc_ids": read_rc_ids, "read_rc_mask": read_rc_mask, "genes": genes}
        buf = []

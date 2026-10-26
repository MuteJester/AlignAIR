"""AA data adapter: GenAIRR productive IGH records -> (tokens, class targets, projected boundary
targets), with explicit rejection accounting.

Reads use GenAIRR's ``sequence_aa`` directly (never re-translated here); only the nucleotide
segment coordinates are projected into AA residue boundaries with straddling-codon masking.
Stop-containing reads are REJECTED, matching the Phase-0 input contract."""
from __future__ import annotations

from alignair.errors import TrainingError

from collections import Counter

import numpy as np
import torch

from .equivalence import build_equivalence_classes, db_hash
from .translation import frame_offset_of, project_codons

REJECT_REASONS = ("not_productive", "no_aa", "no_frame", "stop_codon", "unknown_allele",
                  "no_bounds", "too_long", "empty", "translation_length_mismatch")


class SeedBudgetExhausted(TrainingError):
    """A stream ran out of its allotted seed range before collecting the samples requested."""
    code = "ALIGNAIR_SEED_BUDGET_EXHAUSTED"


_SEG_COLS = {"V": ("v_sequence_start", "v_sequence_end"),
             "D": ("d_sequence_start", "d_sequence_end"),
             "J": ("j_sequence_start", "j_sequence_end")}


def _first(call: str) -> str:
    return str(call).split(",")[0].strip()


def _segments(rec) -> dict:
    segs = {}
    for seg, (a, b) in _SEG_COLS.items():
        s, e = rec.get(a), rec.get(b)
        if s is not None and e is not None:
            segs[seg] = (int(s), int(e))
    return segs


def record_to_sample(rec: dict, class_maps: dict, tokenizer, max_len: int):
    """-> (sample, None) or (None, reason). ``max_len`` is exclusive: ``len(tokens) < max_len`` so
    the exclusive end slot ``n`` exists."""
    if not rec.get("productive"):
        return None, "not_productive"
    aa = rec.get("sequence_aa")
    if not aa:
        return None, "no_aa"
    if rec.get("junction_start") is None:
        return None, "no_frame"
    if "*" in aa:
        return None, "stop_codon"
    frame = frame_offset_of(rec["junction_start"])
    if len(aa) != (len(rec["sequence"]) - frame) // 3:
        # tokens and projected boundaries would live in different coordinate systems
        return None, "translation_length_mismatch"
    try:
        v_class = class_maps["V"].class_of(_first(rec["v_call"]))
        j_class = class_maps["J"].class_of(_first(rec["j_call"]))
    except KeyError:
        return None, "unknown_allele"
    proj = project_codons(len(rec["sequence"]), frame, _segments(rec))
    vb, jb = proj.aa_bounds.get("V"), proj.aa_bounds.get("J")
    if vb is None or jb is None:
        return None, "no_bounds"
    tokens = tokenizer.encode(aa)
    if len(tokens) == 0:
        return None, "empty"
    if len(tokens) >= max_len:                     # reserve slot n for the exclusive endpoint
        return None, "too_long"
    return {"tokens": tokens, "v_class": v_class, "j_class": j_class,
            "v_start": vb[0], "v_end": vb[1], "j_start": jb[0], "j_end": jb[1]}, None


def collate_aa(samples: list, max_len: int, device: str = "cpu"):
    B = len(samples)
    tokens = torch.zeros(B, max_len, dtype=torch.long)
    for i, s in enumerate(samples):
        t = np.asarray(s["tokens"], dtype=np.int64)
        tokens[i, :len(t)] = torch.from_numpy(t)
    batch_in = {"tokenized_sequence": tokens.to(device)}
    genes = ["v", "j"] + (["d"] if "d_class" in samples[0] else [])
    targets = {f"{g}_class": torch.tensor([s[f"{g}_class"] for s in samples], dtype=torch.long, device=device)
               for g in genes}
    for g in genes:
        for b in ("start", "end"):
            targets[f"{g}_{b}"] = torch.tensor([[float(s[f"{g}_{b}"])] for s in samples],
                                               dtype=torch.float32, device=device)
    if "d" in genes:                                          # D validity (0 for inverted/absent -> masked)
        targets["d_valid"] = torch.tensor([float(s.get("d_valid", 1.0)) for s in samples],
                                          dtype=torch.float32, device=device)
    return batch_in, targets


def build_class_maps(dc_name: str = "HUMAN_IGH_OGRDB"):
    """Frozen V/J protein-equivalence class maps from a GenAIRR dataconfig."""
    import GenAIRR as ga

    from .reference import build_aa_germlines
    dc = getattr(ga, dc_name)
    gset = build_aa_germlines(dc, genes=("V", "J"))
    dbh = db_hash([(a.name, a.ungapped_seq) for a in dc.allele_list("v")]
                  + [(a.name, a.ungapped_seq) for a in dc.allele_list("j")])
    class_maps = {g: build_equivalence_classes(
        g, {n: x.aa_seq for n, x in gset.genes[g].items()}, dbh) for g in ("V", "J")}
    meta = {"n_v_classes": class_maps["V"].n_classes(),
            "n_j_classes": class_maps["J"].n_classes(),
            "db_hash": dbh,
            "quarantined_v": len(gset.quarantined["V"]),
            "quarantined_j": len(gset.quarantined["J"])}
    return class_maps, meta


def collect_usable(*, start_seed, budget, want, class_maps, tokenizer, max_len, rate=0.05,
                   dc_name: str = "HUMAN_IGH_OGRDB", counter=None, select=None):
    """Collect ``want`` usable samples from the seed range ``[start_seed, start_seed+budget)``.

    GenAIRR consumes CONSECUTIVE per-record seeds, so a stream seeded ``S`` occupies ``S..S+N-1``
    after N generated records. Ranges must therefore be allocated disjointly and each capped by a
    budget; exhausting the budget raises rather than silently returning a short/overlapping set.

    ``select`` optionally filters/prioritizes (see ``diverse_selector``). Returns
    ``(samples, info)`` where ``info`` records the consumed ``[first_seed, last_seed]``.
    """
    import GenAIRR as ga
    c = counter if counter is not None else Counter()
    exp = ga.Experiment.on(dc_name).recombine().mutate(model="s5f", rate=rate)
    out: list = []
    generated = 0
    for rec in exp.stream_records(n=budget, seed=start_seed):
        generated += 1
        c["generated"] += 1
        s, why = record_to_sample(rec, class_maps, tokenizer, max_len)
        if s is None:
            c[why] += 1
            continue
        c["usable"] += 1
        if select is None or select(s, out):
            out.append(s)
            if len(out) >= want:
                break
    if len(out) < want:
        raise SeedBudgetExhausted(
            f"collected {len(out)}/{want} usable samples after exhausting the {budget}-record "
            f"budget at seeds [{start_seed}, {start_seed + generated - 1}]; widen the budget")
    info = {"start_seed": start_seed, "budget": budget, "generated": generated,
            "first_seed": start_seed, "last_seed": start_seed + generated - 1}
    return out, info


def check_counter(counter) -> None:
    """Every generated record must be accounted for: generated == usable + sum(rejections)."""
    rejected = sum(counter[r] for r in REJECT_REASONS)
    if counter["generated"] != counter["usable"] + rejected:
        raise AssertionError(f"unaccounted records: generated={counter['generated']} "
                             f"usable={counter['usable']} rejected={rejected}")

"""Batching + a gym-backed example stream for AIRRistotle training."""
from __future__ import annotations
import torch
from .prompt import build_example


def _dataconfig_of(reference_set):
    import GenAIRR.data as gdata
    return gdata.HUMAN_IGH_OGRDB                   # MVP: IGH only


def stream_examples(reference_set, tokenizer, params, n, seed, n_distractors=8):
    import random
    from ..gym.gym import build_experiment
    rng = random.Random(seed)
    exp = build_experiment(_dataconfig_of(reference_set), params)
    for rec in exp.stream_records(n=n, seed=seed):
        yield build_example(rec, reference_set, tokenizer, n_distractors=n_distractors, rng=rng)


def collate(examples, pad_id):
    L = max(len(e.input_ids) for e in examples)
    def pad(seq, val): return seq + [val] * (L - len(seq))
    def T(key, val): return torch.tensor([pad(getattr(e, key), val) for e in examples], dtype=torch.long)
    return {
        "input_ids": T("input_ids", pad_id),
        "gen_target": T("gen_target", 0),
        "copy_target": T("copy_target", 0),
        "is_copy": T("is_copy", 0),
        "loss_mask": T("loss_mask", 0),
        "prompt_len": torch.tensor([e.prompt_len for e in examples], dtype=torch.long),
    }

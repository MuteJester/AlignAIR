"""Exponential-moving-average teacher for self-distillation.

The teacher is a slowly-updated copy of the student. The student sees the HARD
view of a recombination (aggressive crop, orientation, errors); the teacher sees
an EASY view (full read, forward) and produces higher-quality soft posteriors that
the student is distilled toward. An EMA target is smoother than the student's own
moving predictions, which stabilises training, and it transfers the teacher's
full-context allele knowledge to the fragment student.
"""
import copy

import torch


class EMATeacher:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, student) -> None:
        """teacher <- decay*teacher + (1-decay)*student (params); buffers copied."""
        d = self.decay
        for tp, sp in zip(self.model.parameters(), student.parameters()):
            tp.mul_(d).add_(sp.detach(), alpha=1.0 - d)
        for tb, sb in zip(self.model.buffers(), student.buffers()):
            tb.copy_(sb)

    @torch.no_grad()
    def to(self, device):
        self.model.to(device)
        return self

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

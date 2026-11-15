"""D presence: whether a read shows its D, from the parser's and the D scan's evidence (inputs detached).

``DPresenceEvidence`` maps four numbers per read, the largest per-base D marginal, the D marginal's sum over
``D_LENGTH_SCALE`` bases, the D posterior's peak and its entropy, to logits over ``D_CLASSES`` (absent, too short to
call, callable).
"""
from __future__ import annotations

from torch import nn
from torch.nn import functional as F

from ..core.d_presence import D_CLASSES

D_LENGTH_SCALE = 50.       # the D marginal's sum is read in units of this many bases


class DPresenceEvidence(nn.Module):
    """(B, 4) evidence -> (B, 3) logits over ``D_CLASSES``: Linear(4, 16), GELU, Linear(16, 3)."""

    def __init__(self):
        super().__init__()
        self.hidden, self.classes = nn.Linear(4, 16), nn.Linear(16, len(D_CLASSES))

    def forward(self, evidence):
        return self.classes(F.gelu(self.hidden(evidence)))

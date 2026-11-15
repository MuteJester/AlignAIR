"""The D presence a model with a D predicts (its D presence head, built exactly when ``has_d``): the one definition of
its classes, its keys and the length a D needs to be called.

A read's D is one of three classes, by its observed germline bases (the D germline from its first to its last aligned
base, internal deletions included, inside the window the model sees), as the AlignAIR assay's truth measures a
segment's observed length:

- ``ABSENT``: the read shows no aligned D base (no D, or one end loss or the window removed);
- ``SHORT``: fewer than ``D_CALLABLE_MIN`` observed germline bases, too short to call (the assay asks only that a
  tool abstain or call it right);
- ``CALLABLE``: at least ``D_CALLABLE_MIN``.

The class gates the D call at prediction: an absent D is dropped, a short one keeps its span without a call.
"""
from __future__ import annotations

ABSENT, SHORT, CALLABLE = range(3)
D_CLASSES = ("absent", "short", "callable")       # class index -> name; the name is the AIRR record's field value
D_CALLABLE_MIN = 8                                # observed germline bases a D needs to be called
D_PRESENCE = "d_presence"                         # target, loss term and weight, record field
D_PRESENCE_LOGITS = "d_presence_logits"           # model output (B, 3)

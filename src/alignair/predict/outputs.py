"""The network's outputs for one batch, typed (``ModelOutputs``): the one place the prediction pipeline reads the
network's output keys.

``ModelOutputs.from_network`` takes ``run_model``'s dict for a batch: each gene the model calls gives its allele scores
and its boundary expectations (``{g}_allele``, ``{g}_start``, ``{g}_end``, required) and its log posterior; the
parser gives boundary logits for every gene it scores (a light chain's D included); the read gives its mutation rate
and indel count (required), and, from the heads a model may have, its orientation, the slot masks, the V and J anchor
logits and the D presence logits; a model bound to a reference of several loci gives each read's locus index
(``read_locus``, ``refine.locus``). A required key missing, a key outside these, or one key of a pair without the other
(a gene's start and end logits, the V and J anchor logits, anchor logits and the start mask) is an InferenceError.
The arrays are kept as given; ``network_dict`` gives the network's dict back.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..core.anchors import ANCHOR_GENES, ANCHOR_LOGITS
from ..core.d_presence import D_PRESENCE_LOGITS
from ..reference.genes import GENES
from ..errors import InferenceError

_READ_KEYS = ("mutation_rate", "indel_count", "orientation", "orientation_logits", "start_mask", "end_mask",
              "read_locus")


@dataclass(frozen=True)
class GeneHeads:
    """One called gene's outputs: its allele scores relative to the read's best allele ``(B, A)``, its boundary
    expectations ``(B, 1)`` and its log posterior ``(B, A)`` (None when the model gives none)."""
    allele: np.ndarray
    start: np.ndarray
    end: np.ndarray
    log_posterior: np.ndarray | None = None


@dataclass(frozen=True)
class ModelOutputs:
    """One batch of the network's outputs (see the module docstring)."""
    genes: Mapping[str, GeneHeads]                                  # the model's genes, in v, d, j order
    boundary_logits: Mapping[str, tuple[np.ndarray, np.ndarray]]    # gene -> (start, end) logits, every parsed gene
    mutation_rate: np.ndarray                                       # (B, 1)
    indel_count: np.ndarray                                         # (B, 1)
    orientation: np.ndarray | None = None
    orientation_logits: np.ndarray | None = None
    start_mask: np.ndarray | None = None
    end_mask: np.ndarray | None = None
    anchor_logits: Mapping[str, np.ndarray] | None = None           # "v", "j" -> (B, L + 1)
    d_presence_logits: np.ndarray | None = None                     # (B, 3)
    read_locus: np.ndarray | None = None                            # (B,) the read's locus index (several loci)

    @classmethod
    def from_network(cls, outputs: Mapping[str, np.ndarray], genes: Sequence[str]) -> "ModelOutputs":
        """The typed outputs of one batch of ``run_model`` for a model calling ``genes`` (see the module docstring)."""
        known = {*(f"{g}_{part}" for g in genes for part in ("allele", "start", "end", "log_posterior")),
                 *(f"{g}_{side}_logits" for g in GENES for side in ("start", "end")),
                 *_READ_KEYS, *ANCHOR_LOGITS.values(), D_PRESENCE_LOGITS}
        unknown = sorted(set(outputs) - known)
        if unknown:
            raise InferenceError(f"model outputs the prediction pipeline does not read: {', '.join(unknown)}",
                                 context={"fields": unknown})
        for key in (*(f"{g}_{part}" for g in genes for part in ("allele", "start", "end")), "mutation_rate",
                    "indel_count"):
            if key not in outputs:
                raise InferenceError(f"model output {key} is missing", context={"field": key})
        pairs = [(f"{g}_start_logits", f"{g}_end_logits") for g in GENES]
        pairs.append(tuple(ANCHOR_LOGITS[g] for g in ANCHOR_GENES))
        for a, b in pairs:
            if (a in outputs) != (b in outputs):
                present, missing = (a, b) if a in outputs else (b, a)
                raise InferenceError(f"model output {present} without {missing}", context={"fields": [present, missing]})
        anchors = None
        if ANCHOR_LOGITS[ANCHOR_GENES[0]] in outputs:
            if "start_mask" not in outputs:
                raise InferenceError("model output anchor logits without start_mask",
                                     context={"fields": [ANCHOR_LOGITS[ANCHOR_GENES[0]], "start_mask"]})
            anchors = {g: outputs[ANCHOR_LOGITS[g]] for g in ANCHOR_GENES}
        return cls(
            genes={g: GeneHeads(outputs[f"{g}_allele"], outputs[f"{g}_start"], outputs[f"{g}_end"],
                                outputs.get(f"{g}_log_posterior")) for g in genes},
            boundary_logits={g: (outputs[f"{g}_start_logits"], outputs[f"{g}_end_logits"]) for g in GENES
                             if f"{g}_start_logits" in outputs},
            mutation_rate=outputs["mutation_rate"], indel_count=outputs["indel_count"],
            orientation=outputs.get("orientation"), orientation_logits=outputs.get("orientation_logits"),
            start_mask=outputs.get("start_mask"), end_mask=outputs.get("end_mask"), anchor_logits=anchors,
            d_presence_logits=outputs.get(D_PRESENCE_LOGITS), read_locus=outputs.get("read_locus"))

    def network_dict(self) -> dict[str, np.ndarray]:
        """The network's dict these outputs were read from: its keys, the same arrays."""
        out: dict[str, np.ndarray] = {}
        for g, heads in self.genes.items():
            out.update({f"{g}_allele": heads.allele, f"{g}_start": heads.start, f"{g}_end": heads.end})
            if heads.log_posterior is not None:
                out[f"{g}_log_posterior"] = heads.log_posterior
        for g, (start, end) in self.boundary_logits.items():
            out.update({f"{g}_start_logits": start, f"{g}_end_logits": end})
        out.update({"mutation_rate": self.mutation_rate, "indel_count": self.indel_count})
        for key in ("orientation", "orientation_logits", "start_mask", "end_mask", "read_locus"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if self.anchor_logits is not None:
            out.update({ANCHOR_LOGITS[g]: logits for g, logits in self.anchor_logits.items()})
        if self.d_presence_logits is not None:
            out[D_PRESENCE_LOGITS] = self.d_presence_logits
        return out

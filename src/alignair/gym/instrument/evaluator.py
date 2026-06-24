"""LatticeEvaluator: run a model over the FrozenLattice and return per-cell
competence (mean + bootstrap CI) on the external CompetenceMetric.

Generation reuses AlignAIRGym with a fixed-params curriculum pinned to the cell's
GenAIRR params; the forward + per-read scoring mirror GymTrainer.evaluate (end-to-end:
predicted region + predicted top-1 allele). Junction exact-match is stubbed to 0 until
junction emission is wired in a later phase (it becomes the junction gate then)."""
import torch
from torch.utils.data import DataLoader

from .. import AlignAIRGym, gym_collate
from ...nn.germline_aligner import decode_germline_coords
from ...training.germline_tf import compute_germline_logits

IGNORE = -100


class _FixedCurriculum:
    """A curriculum that always returns one cell's params (difficulty pinned)."""
    def __init__(self, params: dict):
        self._params = params

    def params(self, p: float = 0.0) -> dict:
        return dict(self._params)

    def describe(self, p: float = 0.0) -> str:
        return "frozen-lattice cell"

    def stage(self, p: float = 0.0) -> int:
        return 0


class LatticeEvaluator:
    def __init__(self, model, reference_set, lattice, metric, dataconfigs,
                 device=None, batch_size: int = 32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.reference_set = reference_set
        self.lattice = lattice
        self.metric = metric
        self.dataconfigs = list(dataconfigs)
        self.batch_size = batch_size
        self.has_d = reference_set.has_d

    def _to_device(self, batch):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    @torch.no_grad()
    def _cell_records(self, cell, n: int) -> list:
        cur = _FixedCurriculum(self.lattice.cell_params(cell))
        gym = AlignAIRGym(self.dataconfigs, self.reference_set, n=n,
                          seed=self.lattice.seed, curriculum=cur)
        loader = DataLoader(gym, batch_size=self.batch_size,
                            collate_fn=lambda b: gym_collate(b, self.reference_set, self.has_d))
        self.model.eval()
        ref_emb = self.model.encode_reference(self.reference_set)
        genes = ["v", "j"] + (["d"] if self.has_d else [])
        records = []
        for batch in loader:
            batch = self._to_device(batch)
            out = self.model(batch["tokens"], batch["mask"], ref_emb,
                             orientation_ids=batch["orientation_id"])
            canon = out["canon_tokens"]
            B = batch["tokens"].shape[0]
            pred_region = out["region_logits"].argmax(-1)
            pred_idx = {g.upper(): out["match"][g.upper()].argmax(-1) for g in genes}
            # end-to-end germline coords: predicted region + predicted top-1 allele
            gl = compute_germline_logits(self.model, canon, batch["mask"], batch, ref_emb,
                                         self.has_d, region_labels=pred_region, allele_idx=pred_idx)
            coords = {}
            for g in genes:
                gs, ge = decode_germline_coords(gl[g][0], gl[g][1], soft=True)
                coords[g] = (gs.cpu(), ge.cpu())
            valid = batch["region_labels"] != IGNORE
            region_match = ((out["region_logits"].argmax(-1) == batch["region_labels"]) & valid)
            reg_num = region_match.sum(dim=1).cpu()
            reg_den = valid.sum(dim=1).clamp(min=1).cpu()
            # D is deliberately unsupervised on inverted-D / D-absent rows (gym_collate
            # zeroes its multi-hot); scoring D there would penalize the model on reads it
            # was never asked to call. Skip the D sub-metrics for those rows.
            d_sup = batch.get("d_supervise")
            for i in range(B):
                # junction_exact intentionally NOT emitted yet (no junction string from
                # the model) -> excluded from competence, not scored 0 (see CompetenceMetric).
                rec = {"region_acc": float(reg_num[i]) / float(reg_den[i]), "coord_errs": []}
                for g in genes:
                    if g == "d" and d_sup is not None and float(d_sup[i]) == 0.0:
                        continue
                    G = g.upper()
                    pi = int(pred_idx[G][i])
                    rec[f"{g}_call_correct"] = int(batch[f"{g}_allele"][i, pi] > 0)
                    gs, ge = coords[g]
                    rec["coord_errs"].append(abs(float(gs[i]) - float(batch[f"{g}_germline_start"][i].cpu())))
                    rec["coord_errs"].append(abs(float(ge[i]) - float(batch[f"{g}_germline_end"][i].cpu())))
                records.append(rec)
        return records

    def eval_cell(self, cell, n: int | None = None) -> dict:
        records = self._cell_records(cell, n if n is not None else cell.n)
        return self.metric.aggregate(records, seed=self.lattice.seed)

    def eval_all(self, n_per_cell: int | None = None) -> dict:
        return {c.name: self.eval_cell(c, n_per_cell) for c in self.lattice.cells}

    def eval_cell_components(self, cell, n: int | None = None) -> dict:
        """Diagnostic: per-cell competence decomposed into allele (reader), coords
        (aligner), region components, each with a bootstrap CI."""
        records = self._cell_records(cell, n if n is not None else cell.n)
        return self.metric.components(records, seed=self.lattice.seed)

    def eval_all_components(self, n_per_cell: int | None = None) -> dict:
        return {c.name: self.eval_cell_components(c, n_per_cell) for c in self.lattice.cells}

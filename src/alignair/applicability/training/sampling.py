"""Exact-size guard batches with a versioned allocation policy and resumable local RNG."""
from __future__ import annotations

import math
import random
from collections.abc import Mapping
from numbers import Integral, Real

from alignair.applicability.tokenizer import enforce_length_contract
from alignair.errors import ArtifactIntegrityError, ConfigurationError, DataContractError, TrainingError, error_context

SAMPLING_POLICY = "alignair.guard.balanced_sampling.v2"
_RNG_SEED_OFFSET = 1000
_ROLE_GROUPS = {
    "positive": ("positive_real", "positive_sim"),
    "biological_negative": ("vhh_nonhuman", "igsf", "igk_igl", "tcr"),
    "structural": ("structural",),
}


def require_sampling_policy(manifest):
    """Refuse execution of an unversioned or different historical training program."""
    if not isinstance(manifest, Mapping) or manifest.get("sampling_policy") != SAMPLING_POLICY:
        raise ArtifactIntegrityError(
            "Guard manifest does not declare the current sampling policy.",
            context={"stage": "guard_training_sampling", "field": "sampling_policy", "expected": SAMPLING_POLICY},
            hint="Freeze a new manifest and use a new checkpoint directory for the corrected policy.",
        )


def _records(records, pool):
    if isinstance(records, (str, bytes, Mapping)):
        raise DataContractError("Sampling pools must contain record mappings.", context={"pool": pool})
    try:
        records = tuple(records)
    except TypeError as exc:
        raise DataContractError("Sampling pools must be iterable.", context={"pool": pool}) from exc
    for index, record in enumerate(records):
        with error_context(pool=pool, record_index=index):
            if not isinstance(record, Mapping) or not isinstance(record.get("aa"), str) or not record["aa"]:
                raise DataContractError("Sampling records require nonempty protein sequence text.", context={"field": "aa"})
            if not enforce_length_contract(record["aa"]).ok:
                raise DataContractError("Sampling records exceed the guard length contract.", context={"field": "aa"})
    return records


class GuardBatchSampler:
    """Sample with replacement; rotate remainder slots across a fixed order of active roles.

    All accepted records fit tensor preparation's length contract. Each nonempty role receives at
    least one slot, role counts differ by at most one, and OE occupies at most its declared cap.
    Pools are snapshotted as tuples; record contents remain caller-owned and must stay unchanged.
    """

    def __init__(self, train, oe, *, batch_size, oe_cap_fraction, seed):
        with error_context(stage="guard_training_sampling"):
            if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
                raise ConfigurationError("Guard batch size must be a positive integer.", context={"field": "batch_size"})
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise ConfigurationError("Sampling seed must be an integer.", context={"field": "seed"})
            try:
                valid = (isinstance(oe_cap_fraction, Real) and not isinstance(oe_cap_fraction, bool)
                         and math.isfinite(oe_cap_fraction) and 0 <= oe_cap_fraction < .5)
            except OverflowError:
                valid = False
            if not valid:
                raise ConfigurationError("OE cap must be finite and within [0, 0.5).", context={"field": "oe_cap_fraction"})
            train, self._oe = _records(train, "train"), _records(oe, "oe")
            pools = {role: [] for role in _ROLE_GROUPS}
            for index, record in enumerate(train):
                group = record.get("group")
                role = next((role for role, groups in _ROLE_GROUPS.items() if group in groups), None) if isinstance(group, str) else None
                if role is None:
                    raise DataContractError("Training record has no recognized sampling role.",
                                            context={"pool": "train", "record_index": index, "field": "group"})
                pools[role].append(record)
            self._pools = {role: tuple(records) for role, records in pools.items() if records}
            if not self._pools:
                raise ConfigurationError("Guard sampling requires nonempty training roles.")
            self._batch_size, self._cap, self._seed = int(batch_size), float(oe_cap_fraction), int(seed)
            try:
                self._n_oe = int(self._batch_size * self._cap) if self._oe else 0
            except OverflowError as exc:
                raise ConfigurationError("Batch size exceeds representable sampling allocation.",
                                         context={"field": "batch_size"}) from exc
            if self._n_oe / self._batch_size > self._cap:
                self._n_oe -= 1  # Keep the cap even when floating multiplication rounds upward.
            main = self._batch_size - self._n_oe
            if main < len(self._pools):
                raise ConfigurationError("Batch size cannot cover every active role after reserving the OE quota.",
                                         context={"field": "batch_size", "active_roles": len(self._pools)})
            self._base, self._remainder = divmod(main, len(self._pools))
            self._rng = random.Random(self._seed + _RNG_SEED_OFFSET)
            self._batches = 0

    def plan(self):
        """Return JSON-compatible policy, allocation and pool-count provenance."""
        return {"policy": SAMPLING_POLICY, "batch_size": self._batch_size, "oe_cap_fraction": self._cap,
                "oe_per_batch": self._n_oe, "oe_pool_size": len(self._oe), "seed": self._seed,
                "rng_seed": self._seed + _RNG_SEED_OFFSET,
                "role_order": list(self._pools), "role_pool_sizes": {role: len(pool) for role, pool in self._pools.items()},
                "base_per_role": self._base, "remainder_slots": self._remainder,
                "remainder_allocation": "rotate_one_role_per_batch", "replacement": True}

    def draw(self):
        """Return separate main/OE record lists, totaling the configured batch size."""
        roles = tuple(self._pools)
        extras = {(self._batches + offset) % len(roles) for offset in range(self._remainder)}
        main = [self._rng.choice(self._pools[role]) for index, role in enumerate(roles)
                for _ in range(self._base + (index in extras))]
        oe = [self._rng.choice(self._oe) for _ in range(self._n_oe)]
        self._batches += 1
        return main, oe

    def state_dict(self):
        return {"policy": SAMPLING_POLICY, "plan": self.plan(), "batches_drawn": self._batches, "rng": self._rng.getstate()}

    def load_state_dict(self, state, *, expected_batches=None):
        """Restore counter and RNG together; malformed state leaves this sampler unchanged."""
        previous = self._rng.getstate()
        try:
            if not isinstance(state, Mapping) or state.get("policy") != SAMPLING_POLICY:
                raise ValueError("Sampling state policy differs or is absent")
            if state.get("plan") != self.plan():
                raise ValueError("Sampling state allocation differs")
            batches = state.get("batches_drawn")
            if isinstance(batches, bool) or not isinstance(batches, int) or batches < 0:
                raise ValueError("Invalid sampling batch counter")
            if expected_batches is not None and batches != expected_batches:
                raise ValueError("Sampling counter differs from checkpoint progress")
            self._rng.setstate(state["rng"])
        except (KeyError, ValueError, TypeError, IndexError, OverflowError) as exc:
            self._rng.setstate(previous)
            raise TrainingError("guard checkpoint sampling RNG state cannot be restored",
                                context={"stage": "guard_training_resume_rng", "field": "sampling_rng"}) from exc
        self._batches = batches

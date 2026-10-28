"""Preflight convergence study requests and scoring inputs before expensive work."""
from collections.abc import Mapping, Sequence
from numbers import Integral
import math

import numpy as np
import torch

from alignair.calibration import Conformal
from alignair.core import AlignAIRConfig
from alignair.core.setup import _input_config
from alignair.data.inference import SequenceInputAdapter
from alignair.data.sampling import ProteinSampleBuilder
from alignair.data.tokenizer import AAProteinTokenizer, AA_TOKEN_DICT
from alignair.errors import ConfigurationError, InputError, error_context
from alignair.reference import EquivalenceClassMap
from alignair.evaluate.probabilities import validate_frequency_edges
from alignair.train.convergence.cohort import _first


def validate_study_request(budget, ckpt_steps, cohort_seed, n_per_stratum, batch_size,
                           base_lr, lr_min, val_every, plateau_k, dc_name, device):
    """Check continuation options before checkpoint loading, allocation or output creation."""
    from alignair.train.schedules import CosineSchedule
    with error_context(stage="convergence_study_preflight"):
        for field, value, minimum in (("budget", budget, 1), ("cohort_seed", cohort_seed, 0),
                ("n_per_stratum", n_per_stratum, 1), ("batch_size", batch_size, 1),
                ("val_every", val_every, 0), ("plateau_k", plateau_k, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ConfigurationError("study counts must be valid integers", context={"field": field})
        if not isinstance(dc_name, str) or not dc_name:
            raise ConfigurationError("study requires a dataconfig name", context={"field": "dc_name"})
        if not isinstance(device, str) or not device:
            raise ConfigurationError("study device must be a nonempty string", context={"field": "device"})
        try:
            torch.device(device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ConfigurationError("invalid study device", context={"field": "device"}) from exc
        for field, value in (("base_lr", base_lr), ("lr_min", lr_min)):
            try:
                valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            except OverflowError:
                valid = False
            if not valid or field == "base_lr" and value == 0:
                raise ConfigurationError("learning rates must be finite; base_lr positive and lr_min nonnegative",
                                         context={"field": field})
        CosineSchedule(budget, lr_min).validate(base_lr)
        if isinstance(ckpt_steps, (str, bytes, Mapping, set, frozenset)):
            raise ConfigurationError("checkpoint offsets must be an ordered integer iterable", context={"field": "ckpt_steps"})
        try:
            offsets = tuple(ckpt_steps)
        except TypeError as exc:
            raise ConfigurationError("checkpoint offsets must be iterable", context={"field": "ckpt_steps"}) from exc
        if (any(isinstance(step, bool) or not isinstance(step, int) or not 0 < step <= budget for step in offsets)
                or any(left >= right for left, right in zip(offsets, offsets[1:]))):
            raise ConfigurationError("checkpoint offsets must be strictly increasing integers within the study budget",
                                     context={"field": "ckpt_steps"})
        return offsets


def _rows(value, field):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise InputError("cohort rows must be frozen indexable sequences", context={"field": field})
    return value


def prepare_evaluation(model, config, class_maps, meta, cohort, device, alpha, freq_edges):
    """Check consumed inputs and prepare boundary samples with the shared acceptance policy.

    This does not move/forward the model, change its modes, mutate records, or verify artifact
    identity. A custom torch module without cfg is allowed; column identity remains caller-owned.
    """
    with error_context(stage="convergence_preflight"):
        if not isinstance(model, torch.nn.Module):
            raise ConfigurationError("checkpoint scoring requires a torch module", context={"field": "model"})
        if (not isinstance(config, AlignAIRConfig) or config.input_space != "protein"
                or config.label_space != "protein_classes" or config.classification_mode != "categorical"
                or config.boundary_mode != "half_open" or type(config.has_d) is not bool
                or not config.has_d and config.d_allele_count != 0):
            raise ConfigurationError("convergence scoring requires a categorical protein-class configuration",
                                     context={"field": "config"})
        if getattr(model, "cfg", config) != config:
            raise ConfigurationError("scoring configuration differs from the model configuration", context={"field": "config"})
        _input_config(config, "protein")
        Conformal(alpha=alpha)  # Reuse the calibration owner's parameter contract, even for an empty slice.
        validate_frequency_edges(freq_edges)
        if not isinstance(class_maps, Mapping):
            raise ConfigurationError("scoring requires V/J equivalence maps", context={"field": "class_maps"})
        for gene in ("V", "J"):
            labels = class_maps.get(gene)
            count = getattr(config, f"{gene.lower()}_allele_count")
            if (not isinstance(labels, EquivalenceClassMap) or labels.gene != gene
                    or isinstance(count, bool) or not isinstance(count, Integral) or count <= 0):
                raise ConfigurationError("scoring requires valid configured V/J equivalence maps", context={"gene": gene})
            labels.validate()
            if labels.n_classes() != count:
                raise ConfigurationError("class map width differs from the configured head", context={"gene": gene})
        if not isinstance(meta, Mapping):
            raise ConfigurationError("scoring metadata must be a mapping", context={"field": "meta"})
        d_index = meta.get("d_index") if config.has_d else None
        if config.has_d and (not isinstance(d_index, Mapping) or not d_index
                or isinstance(config.d_allele_count, bool) or not isinstance(config.d_allele_count, Integral)
                or config.d_allele_count != len(d_index)):
            raise ConfigurationError("D vocabulary width differs from the configured head", context={"field": "d_index"})
        builder = ProteinSampleBuilder(class_maps, AAProteinTokenizer(), config.max_seq_length, d_index=d_index)
        adapter = SequenceInputAdapter(AAProteinTokenizer(), config.max_seq_length, device,
                                        alphabet=frozenset(AA_TOKEN_DICT), vocab_size=config.vocab_size)
        if (not isinstance(cohort, Mapping) or not isinstance(cohort.get("by_stratum"), Mapping)
                or not isinstance(cohort.get("dc_name"), str) or not cohort["dc_name"]):
            raise InputError("cohort requires named strata and a dataconfig name", context={"field": "cohort"})
        strata = cohort["by_stratum"]
        if "clean" not in strata or not _rows(strata["clean"], "clean"):
            raise InputError("evaluate_checkpoint: cohort has no 'clean' stratum rows", context={"field": "clean"})
        calibration = _rows(cohort.get("calibration"), "calibration")
        for name, rows in (*strata.items(), ("__calibration__", calibration)):
            if not isinstance(name, str) or not name:
                raise InputError("stratum names must be nonempty strings", context={"field": "stratum"})
            with error_context(stratum=name):
                rows = _rows(rows, name)
                for index, row in enumerate(rows):
                    with error_context(record_index=index):
                        if not isinstance(row, Mapping):
                            raise InputError("cohort rows must be mappings")
                        for gene in (("V", "J") if name == "clean" else ("V",)):
                            call = row.get(f"{gene.lower()}_call")
                            if not isinstance(call, str) or _first(call) not in class_maps[gene].allele_class:
                                raise InputError("first truth call is absent from the scoring vocabulary", context={"gene": gene})
                        if name == "clean" and config.has_d and (
                                not isinstance(row.get("d_inverted"), (bool, np.bool_))
                                or not row["d_inverted"] and not isinstance(row.get("d_call"), str)):
                            raise InputError("D scoring requires a boolean inversion flag and a string call", context={"gene": "D"})
                adapter.validate([row.get("aa") for row in rows])
        samples = []
        for index, row in enumerate(strata["clean"]):
            with error_context(stratum="clean", record_index=index):
                record = row.get("record")
                if record is None:
                    continue
                sample = builder(record).sample
                if sample is not None:
                    if (record["sequence_aa"] != row["aa"] or any(
                            _first(record.get(f"{gene}_call")) != _first(row[f"{gene}_call"])
                            for gene in ("v", "j"))):
                        raise InputError("boundary source differs from the clean classification row", context={"field": "record"})
                    samples.append(sample)
        return samples

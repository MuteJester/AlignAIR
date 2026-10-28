"""Preflight the convergence scoring contract before model execution or placement."""
from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np
import torch

from ...calibration import Conformal
from ...core import AlignAIRConfig
from ...core.setup import _input_config
from ...data.inference import SequenceInputAdapter
from ...data.sampling import ProteinSampleBuilder
from ...data.tokenizer import AAProteinTokenizer, AA_TOKEN_DICT
from ...errors import ConfigurationError, InputError, error_context
from ...reference import EquivalenceClassMap
from .cohort import _first


def _rows(value, field):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise InputError("cohort rows must be frozen indexable sequences", context={"field": field})
    return value


def _frequency_edges(edges):
    try:
        values = np.asarray(edges)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("frequency edges must form a numeric vector", context={"field": "freq_edges"}) from exc
    if (values.ndim != 1 or len(values) < 2 or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all() or np.any(np.diff(values.astype(float)) <= 0)
            or values[0] < 0 or values[-1] > 1):
        raise ConfigurationError("frequency edges must be finite, strictly increasing and within [0, 1]",
                                 context={"field": "freq_edges"})


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
        _frequency_edges(freq_edges)
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

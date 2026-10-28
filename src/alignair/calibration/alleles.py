"""Fit raw allele-head confidence temperatures over caller-owned held-out records."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..errors import ConfigurationError, InputError, InferenceError, error_context
from .temperature import _count, _probabilities, apply_temperature, ece, fit_temperature


def calibrate_allele_temperatures(model, reference, val_records, genes=None, device="cpu", *,
                                  batch_size=128, min_samples=50) -> tuple[dict, dict]:
    """Fit per-gene sigmoid scaling for either input alphabet without changing model/config.

    Records supply sequence (nucleotide) or sequence_aa (protein), plus comma-separated gene
    calls. Missing calls are unobserved. Store only top-1 confidences/correctness, not all logits.
    The report's acc is raw top-1 accuracy; clipping/rounding and set selection retain their own
    prediction semantics. Temperatures and report field names preserve the historical schema.
    """
    from ..data.inference import SequenceInputAdapter
    from ..data.inputs import prepare_nucleotide_inputs
    from ..data.tokenizer import AAProteinTokenizer, AA_TOKEN_DICT, CenterPaddedTokenizer
    from ..predict.forward import output_numpy, prediction_batches

    batch_size, min_samples = _count(batch_size, "batch_size"), _count(min_samples, "min_samples")
    config = getattr(model, "cfg", None)
    if config is None or getattr(config, "classification_mode", "multilabel") != "multilabel":
        raise ConfigurationError("allele temperature fitting requires a model with sigmoid allele heads")
    input_space = getattr(config, "input_space", "nucleotide")
    if input_space not in ("nucleotide", "protein"):
        raise ConfigurationError("temperature fitting requires nucleotide or protein model inputs")
    specs = getattr(config, "gene_specs", ())
    configured = {spec.name: spec.allele_count for spec in specs} or dict.fromkeys(
        ("v", "d", "j") if getattr(config, "has_d", True) else ("v", "j"))
    try:
        selected = tuple(configured) if genes is None else tuple(genes)
        if isinstance(genes, str) or not selected or len(set(selected)) != len(selected) or not set(selected) <= configured.keys():
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("temperature genes must be a nonempty unique subset of configured genes") from exc
    names = {}
    for gene in selected:
        try:
            names[gene] = tuple(reference.gene(gene.upper()).names)
        except (AttributeError, KeyError, TypeError) as exc:
            raise ConfigurationError("temperature fitting requires the configured reference allele names") from exc
        if (not names[gene] or any(not isinstance(name, str) or not name for name in names[gene])
                or len(set(names[gene])) != len(names[gene])
                or configured[gene] is not None and configured[gene] != len(names[gene])):
            raise ConfigurationError("temperature reference vocabulary differs from the configured allele head",
                                     context={"gene": gene})
    if isinstance(val_records, (str, bytes, Mapping)):
        raise InputError("supply a cohort of temperature calibration records")
    try:
        records = tuple(val_records)
    except TypeError as exc:
        raise InputError("temperature calibration records must be iterable") from exc
    if not records:
        return {}, {}
    field = "sequence" if input_space == "nucleotide" else "sequence_aa"
    sequences, truth = [], {gene: [] for gene in selected}
    for index, record in enumerate(records):
        with error_context(stage="allele_temperature_calibration", record_index=index):
            if not isinstance(record, Mapping):
                raise InputError("temperature calibration records must be mappings")
            sequences.append(record.get(field))
            for gene in selected:
                call = record.get(f"{gene}_call")
                if call is not None and not isinstance(call, str):
                    raise InputError("temperature truth calls must be strings or absent", context={"gene": gene})
                truth[gene].append({name.strip() for name in (call or "").split(",") if name.strip()})
    if input_space == "nucleotide":
        sequences, _ = prepare_nucleotide_inputs(sequences, config.max_seq_length)
        adapter = SequenceInputAdapter(CenterPaddedTokenizer(), config.max_seq_length, device, vocab_size=6)
    else:
        adapter = SequenceInputAdapter(AAProteinTokenizer(), config.max_seq_length, device,
                                        alphabet=frozenset(AA_TOKEN_DICT), vocab_size=getattr(config, "vocab_size", None))
        sequences = adapter.validate(sequences)
    confidence, correct = {gene: [] for gene in selected}, {gene: [] for gene in selected}
    for batch in prediction_batches(model, sequences, adapter, batch_size=batch_size,
                                     output_keys=tuple(f"{gene}_allele" for gene in selected)):
        for gene in selected:
            with error_context(stage="allele_temperature_calibration", gene=gene, record_offset=batch.record_offset):
                values = _probabilities(output_numpy(batch.outputs[f"{gene}_allele"]))
                if values.shape != (batch.size, len(names[gene])):
                    raise InferenceError("allele probability width differs from the temperature reference")
                top = values.argmax(1)
                for local, class_id in enumerate(top):
                    calls = truth[gene][batch.record_offset + local]
                    if calls:
                        confidence[gene].append(float(values[local, class_id]))
                        correct[gene].append(names[gene][class_id] in calls)
    temperatures, report = {}, {}
    for gene in selected:
        if len(confidence[gene]) < min_samples:
            continue
        probabilities, labels = np.asarray(confidence[gene]), np.asarray(correct[gene], dtype=np.float64)
        temperature = fit_temperature(probabilities, labels)
        temperatures[gene] = temperature
        report[gene] = {"T": round(temperature, 4), "ece_before": round(ece(probabilities, labels), 4),
                        "ece_after": round(ece(apply_temperature(probabilities, temperature), labels), 4),
                        "acc": round(float(labels.mean()), 4), "n": len(probabilities)}
    return temperatures, report

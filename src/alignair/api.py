"""User-facing façade for AlignAIR — load a trained model and run predictions (or train a custom
model) in a few lines, without touching the internal config/pipeline machinery.

    from alignair import load_model, predict_sequences
    model, reference = load_model("model.alignair")
    records = predict_sequences(model, reference, ["CAGGTGCAGCTG..."])
    # records: AIRR-convention dicts (v_call/d_call/j_call, coordinates, CIGAR, productivity, ...)

    from alignair import train_model
    train_model(["HUMAN_IGH_OGRDB"], out_path="my_model.alignair", steps=100_000)
"""
from __future__ import annotations

from typing import Sequence

from .errors import InputError

from .core import AlignAIR
from .predict import PredictConfig, predict as _predict
from .reference.reference_set import ReferenceSet

__all__ = ["load_model", "predict_sequences", "train_model"]

def load_model(checkpoint_path: str, *, reference: ReferenceSet | None = None,
               device: str = "cpu") -> tuple[AlignAIR, ReferenceSet]:
    """Load a trained AlignAIR model file into an eval-ready ``(model, reference)`` pair.

    A ``.alignair`` model file carries its own reference (rebuilt from the no-pickle ``reference_json``);
    ``reference`` binds another one to a reference-agnostic model (alleles it never trained on included). An
    AlignAIR 3 checkpoint (a ``.pt``, or a conv-v1 model file) is refused with the version that reads it.
    """
    from .model_file import container, load_model as _load_alignair, v3_checkpoint_error
    if not container.is_alignair_file(checkpoint_path):
        raise v3_checkpoint_error(checkpoint_path)
    lm = _load_alignair(checkpoint_path, device=device, reference=reference)
    return lm.model, lm.reference


def _genotype_bank(reference: ReferenceSet, genotype) -> ReferenceSet:
    """The genotype's subset of ``reference`` ({gene letter: allele names}; a gene it does not name keeps all its
    alleles). InputError naming the alleles the reference does not hold."""
    missing = sorted(name for gene, names in genotype.items() for name in names
                     if gene.upper() not in reference.genes or name not in reference.gene(gene.upper()).index)
    if missing:
        raise InputError(f"the genotype names alleles the reference does not hold: {', '.join(missing[:5])}",
                         context={"missing": missing},
                         hint="Give their sequences in the reference (alignair predict --reference).")
    return reference.subset({gene.upper(): names for gene, names in genotype.items()})


def predict_sequences(model: AlignAIR, reference: ReferenceSet, sequences: Sequence[str], *,
                      device: str | None = None, batch_size: int = 64, **predict_overrides) -> list:
    """Run the full AlignAIR prediction pipeline on raw nucleotide strings, returning AIRR-convention
    records. ``predict_overrides`` forwards to :class:`~alignair.predict.PredictConfig` (e.g.
    ``threshold=``, ``germline_reader=``, ``genotype=``). On a reference-agnostic model a genotype is
    the bank: the model is bound to the genotype's subset of ``reference`` for the call (its calls are
    then only the genotype's alleles, the model reading no other) and to ``reference`` again after."""
    genotype = predict_overrides.get("genotype")
    if genotype and getattr(model, "reference_agnostic", False):
        bank = _genotype_bank(reference, genotype)
        model.bind_reference(bank)
        try:
            return predict_sequences(model, bank, sequences, device=device, batch_size=batch_size,
                                     **{**predict_overrides, "genotype": None})
        finally:
            model.bind_reference(reference)
    device = device or next(model.parameters()).device.type
    cfg = model.cfg
    # the reference's ordered locus schema: one locus labels each record's locus; None for a schema-less reference.
    chain_types = reference.locus_names() or None if hasattr(reference, "locus_names") else None
    predict_overrides.setdefault("chain_types", chain_types)
    pcfg = PredictConfig(max_seq_length=cfg.max_seq_length, has_d=cfg.has_d, batch_size=batch_size,
                         **predict_overrides)
    return _predict(model, sequences, reference, pcfg, device=device)


def train_model(dataconfigs, *, out_path: str, steps: int = 100_000, device: str = "cpu",
                **train_overrides) -> str:
    """Train a new AlignAIR model on one or more GenAIRR dataconfigs and save a resumable checkpoint to ``out_path``.
    ``train_overrides`` forwards to :func:`alignair.train.trainer.train` (e.g. ``lr=``, ``batch_size=``,
    ``progresses=``, ``short_boost=``, ``resume_path=``). Like ``alignair train``, the curriculum's reads get library
    flanks (``data.flanks.FlankPolicy``) unless ``flanks=`` says otherwise (None: bare reads)."""
    from .data.flanks import FlankPolicy
    from .genairr import resolve_dataconfigs
    from .tasks import build_tasks
    from .core import prepare_model
    from .train.guards import validate_training_request
    from .train.trainer import train as _train

    dcs = resolve_dataconfigs(dataconfigs)
    reference = ReferenceSet.from_dataconfigs(*dcs)
    setup = prepare_model(reference)
    cfg = setup.config
    # validate the request BEFORE allocating the (potentially large) model
    validate_training_request(steps=steps, batch_size=train_overrides.get("batch_size", 32),
                              lr=train_overrides.get("lr", 3e-4), max_seq_length=cfg.max_seq_length,
                              reference=reference, progresses=train_overrides.get("progresses", (0.3,)),
                              grad_clip=train_overrides.get("grad_clip"))
    model = setup.build_model()
    tasks = train_overrides.get("tasks")
    tasks = build_tasks(cfg) if tasks is None else tasks
    train_overrides["tasks"] = tasks
    train_overrides.setdefault("flanks", FlankPolicy())
    _train(model, reference, dcs, cfg, steps=steps, device=device,
           save_path=out_path, **train_overrides)
    return out_path

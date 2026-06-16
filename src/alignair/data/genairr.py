"""GenAIRR 2.2.0 helpers. Verify capability, not the (mislabelled) version string."""


def assert_genairr_capable() -> None:
    """Ensure the GenAIRR fluent 2.x Experiment API is importable.

    The local GenAIRR build reports __version__ == '1.0.0' but is actually the
    2.2.0 codebase; we check for the capability (stream_records) instead.
    """
    from GenAIRR import Experiment
    if not hasattr(Experiment, "stream_records"):
        raise RuntimeError(
            "GenAIRR >= 2.2.0 required: Experiment.stream_records is missing")


def allele_vocab_from_dataconfig(dataconfig) -> dict:
    """Per-gene allele vocabulary from a GenAIRR DataConfig.

    D vocabulary is sorted unique names + 'Short-D' as the LAST entry (matching
    the encoder/loss convention that the last D column is the Short-D class).
    """
    vocab = {
        "V": sorted(a.name for a in dataconfig.allele_list("v")),
        "J": sorted(a.name for a in dataconfig.allele_list("j")),
    }
    if dataconfig.metadata.has_d:
        vocab["D"] = sorted(a.name for a in dataconfig.allele_list("d")) + ["Short-D"]
    return vocab

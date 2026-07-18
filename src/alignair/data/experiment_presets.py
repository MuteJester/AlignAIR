"""GenAIRR experiment presets (compiled, ready to stream_records)."""
from .genairr import assert_genairr_capable


def _base(dataconfig):
    from GenAIRR import Experiment
    assert_genairr_capable()
    return Experiment.on(dataconfig).recombine()


def minimal(dataconfig):
    """Recombination + SHM only — no corruption. Returns a compiled experiment."""
    return _base(dataconfig).mutate(model="s5f", rate=0.05).compile()


def no_corruption(dataconfig):
    """Alias of minimal for clarity at call sites."""
    return minimal(dataconfig)


def full_augmentation(dataconfig, *, mutation_rate: float = 0.05, invert_d_prob: float = 0.05,
                      end_loss_5=(0, 25), end_loss_3=(0, 25), indel_count=(0, 5),
                      seq_error_rate: float = 0.001, ambiguous_count=(0, 5)):
    """Legacy-style full augmentation: SHM + 5'/3' loss + indels + sequencing
    errors + ambiguous bases (+ D-inversion if the chain has a D gene).

    Returns a compiled experiment whose ``stream_records`` yields AIRR dicts.
    """
    exp = _base(dataconfig).mutate(model="s5f", rate=mutation_rate)
    if dataconfig.metadata.has_d:
        exp = exp.invert_d(prob=invert_d_prob)
    exp = (exp.end_loss_5prime(length=end_loss_5)
              .end_loss_3prime(length=end_loss_3)
              .polymerase_indels(count=indel_count)
              .sequencing_errors(rate=seq_error_rate)
              .ambiguous_base_calls(count=ambiguous_count))
    return exp.compile()

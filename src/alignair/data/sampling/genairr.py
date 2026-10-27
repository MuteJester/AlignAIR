"""GenAIRR provider adapter for bounded protein-class sample collection."""
from operator import index

from ...errors import ConfigurationError, error_context
from .collection import collect_samples, check_collection_counts, _positive
from .protein import REJECT_REASONS, ProteinSampleBuilder


def collect_usable(*, start_seed, budget, want, class_maps, tokenizer, max_len, rate=.05,
                   dc_name="HUMAN_IGH_OGRDB", counter=None, select=None):
    """Compatibility GenAIRR provider adapter; source intervals count every generated record."""
    import GenAIRR as ga
    want, budget = _positive(want, "want"), _positive(budget, "budget")
    try:
        if isinstance(start_seed, bool) or index(start_seed) < 0:
            raise ValueError
        start_seed = index(start_seed)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("start_seed must be a nonnegative integer") from exc
    builder = ProteinSampleBuilder(class_maps, tokenizer, max_len)
    experiment = ga.Experiment.on(dc_name).recombine().mutate(model="s5f", rate=rate)
    records = experiment.stream_records(n=budget, seed=start_seed)
    with error_context(source=dc_name, start_seed=start_seed):
        result = collect_samples(records, builder, want=want, budget=budget, counter=counter, select=select,
                                 rejection_reasons=REJECT_REASONS)
    info = {"start_seed": start_seed, "budget": budget, "generated": result.generated,
            "first_seed": start_seed, "last_seed": start_seed + result.generated - 1}
    return result.samples, info


def check_counter(counter) -> None:
    """Historical protein-rejection census, delegated to the shared ledger validator."""
    check_collection_counts(counter, REJECT_REASONS)

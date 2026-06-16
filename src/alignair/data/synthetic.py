"""On-the-fly GenAIRR synthetic dataset producing the AlignAIR (x, y) contract."""
import logging

from torch.utils.data import IterableDataset

from .tokenizer import CenterPaddedTokenizer
from .encoders import AlleleEncoder
from .record_adapter import RecordAdapter
from .sample_builder import build_xy

logger = logging.getLogger(__name__)


class SyntheticDataset(IterableDataset):
    """Stream GenAIRR records and adapt them to per-sample (x, y).

    Parameters
    ----------
    compiled_experiment : object with ``stream_records(n, seed)`` (from a preset).
    max_seq_length : int — sequences longer than this are dropped (logged).
    has_d : bool
    allele_vocab : dict {"V": [...], "J": [...], "D": [...]} (D ends with 'Short-D').
    n : int | None — records per epoch (None = infinite stream).
    seed : int — base seed; each __iter__ uses seed + epoch for fresh-but-reproducible data.

    Use with ``num_workers=0`` (single-process); worker sharding is not implemented.
    """

    def __init__(self, compiled_experiment, max_seq_length: int, has_d: bool,
                 allele_vocab: dict, n: int | None = None, seed: int = 0):
        self.experiment = compiled_experiment
        self.max_seq_length = max_seq_length
        self.has_d = has_d
        self.n = n
        self.seed = seed
        self._epoch = 0

        self.tokenizer = CenterPaddedTokenizer(max_length=max_seq_length)
        self.adapter = RecordAdapter(has_d=has_d)
        self.encoder = AlleleEncoder()
        for gene in (["V", "J"] + (["D"] if has_d else [])):
            self.encoder.register_gene(gene, allele_vocab[gene], sort=False)

    def __iter__(self):
        seed = self.seed + self._epoch
        self._epoch += 1
        dropped = 0
        for record in self.experiment.stream_records(n=self.n, seed=seed):
            seq = str(record["sequence"]).upper()
            if len(seq) > self.max_seq_length:
                dropped += 1
                continue
            tokens, pad_left = self.tokenizer.encode_and_pad(seq)
            row = dict(record)
            row["indels"] = record.get("n_indels", 0)
            if self.has_d and not row.get("d_call"):
                row["d_call"] = "Short-D"
            rec = self.adapter.adapt(row, pad_left)
            yield build_xy(tokens, rec, self.encoder, self.has_d)
        if dropped:
            logger.warning("SyntheticDataset dropped %d over-length sequences", dropped)

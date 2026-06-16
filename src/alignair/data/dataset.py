"""Map-style CSV dataset producing the AlignAIR (x, y) contract."""
import numpy as np
from torch.utils.data import Dataset

from .column_schema import ColumnSet
from .readers import CsvTableReader
from .tokenizer import CenterPaddedTokenizer
from .encoders import AlleleEncoder
from .record_adapter import RecordAdapter


def allele_vocab_from_csv(csv_path: str, has_d: bool, sep: str = ",") -> dict:
    """Scan a CSV's call columns to build the per-gene allele vocabulary.

    D vocabulary is sorted unique calls + 'Short-D' as the LAST entry (the loss's
    short-D penalty reads the last D column).
    """
    reader = CsvTableReader(csv_path, ColumnSet(has_d=has_d), sep=sep)
    genes = {"V": "v_call", "J": "j_call"}
    if has_d:
        genes["D"] = "d_call"
    vocab: dict = {}
    for gene, col in genes.items():
        seen = set()
        for row in reader.records:
            for a in str(row[col]).split(","):
                if a:
                    seen.add(a)
        if gene == "D":
            seen.discard("Short-D")
            ordered = sorted(seen) + ["Short-D"]
        else:
            ordered = sorted(seen)
        vocab[gene] = ordered
    return vocab


class AlignAIRDataset(Dataset):
    def __init__(self, csv_path: str, max_seq_length: int, has_d: bool,
                 allele_vocab: dict | None = None, sep: str = ",",
                 nrows: int | None = None):
        self.has_d = has_d
        self.max_seq_length = max_seq_length
        self.reader = CsvTableReader(csv_path, ColumnSet(has_d=has_d), sep=sep, nrows=nrows)
        self.tokenizer = CenterPaddedTokenizer(max_length=max_seq_length)
        self.adapter = RecordAdapter(has_d=has_d)

        if allele_vocab is None:
            allele_vocab = allele_vocab_from_csv(csv_path, has_d=has_d, sep=sep)
        self.encoder = AlleleEncoder()
        for gene in (["V", "J"] + (["D"] if has_d else [])):
            self.encoder.register_gene(gene, allele_vocab[gene], sort=False)

        self.v_allele_count = self.encoder.count("V")
        self.j_allele_count = self.encoder.count("J")
        self.d_allele_count = self.encoder.count("D") if has_d else None

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, i: int):
        row = self.reader[i]
        tokens, pad_left = self.tokenizer.encode_and_pad(row["sequence"])
        rec = self.adapter.adapt(row, pad_left)

        x = {"tokenized_sequence": tokens}
        y = {
            "v_start": np.array([rec["v_start"]], np.float32),
            "v_end": np.array([rec["v_end"]], np.float32),
            "j_start": np.array([rec["j_start"]], np.float32),
            "j_end": np.array([rec["j_end"]], np.float32),
            "v_allele": self.encoder.encode("V", [rec["v_call_set"]])[0],
            "j_allele": self.encoder.encode("J", [rec["j_call_set"]])[0],
            "mutation_rate": np.array([rec["mutation_rate"]], np.float32),
            "indel_count": np.array([rec["indel_count"]], np.float32),
            "productive": np.array([rec["productive"]], np.float32),
        }
        if self.has_d:
            y["d_start"] = np.array([rec["d_start"]], np.float32)
            y["d_end"] = np.array([rec["d_end"]], np.float32)
            y["d_allele"] = self.encoder.encode("D", [rec["d_call_set"]])[0]
        return x, y

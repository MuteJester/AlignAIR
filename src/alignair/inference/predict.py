"""End-to-end call prediction: sequences -> allele calls + corrected coordinates."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..data.tokenizer import CenterPaddedTokenizer
from ..data.encoders import AlleleEncoder
from ..postprocessing.allele_selector import select_alleles
from .predictor import Predictor
from .decode import extract_positions, correct_segments


@dataclass
class PredictionResult:
    v_calls: List[List[str]]
    j_calls: List[List[str]]
    d_calls: Optional[List[List[str]]]
    v_start: np.ndarray
    v_end: np.ndarray
    j_start: np.ndarray
    j_end: np.ndarray
    d_start: Optional[np.ndarray]
    d_end: Optional[np.ndarray]
    mutation_rate: np.ndarray
    indel_count: np.ndarray
    productive: np.ndarray


def predict_calls(model, sequences, *, allele_vocab: dict, max_seq_length: int,
                  percentage: float = 0.21, cap: int = 3, batch_size: int = 256) -> PredictionResult:
    has_d = model.config.has_d_gene
    tokenizer = CenterPaddedTokenizer(max_length=max_seq_length)
    tokens = np.stack([tokenizer.encode_and_pad(s.upper())[0] for s in sequences])

    pred = Predictor(model).predict(tokens, batch_size=batch_size)

    positions = extract_positions(pred, has_d)
    corrected = correct_segments(positions, sequences, max_seq_length, has_d)

    encoder = AlleleEncoder()
    for gene in (["V", "J"] + (["D"] if has_d else [])):
        encoder.register_gene(gene, allele_vocab[gene], sort=False)

    def names(gene_key, gene):
        i2a = encoder.gene_encodings[gene].index_to_allele
        return [calls for calls, _lik in select_alleles(pred[gene_key], i2a, percentage, cap)]

    productive = (np.squeeze(pred["productive"], -1) > 0.5)
    return PredictionResult(
        v_calls=names("v_allele", "V"),
        j_calls=names("j_allele", "J"),
        d_calls=names("d_allele", "D") if has_d else None,
        v_start=corrected["v_start"], v_end=corrected["v_end"],
        j_start=corrected["j_start"], j_end=corrected["j_end"],
        d_start=corrected.get("d_start"), d_end=corrected.get("d_end"),
        mutation_rate=np.squeeze(pred["mutation_rate"], -1),
        indel_count=np.squeeze(pred["indel_count"], -1),
        productive=productive,
    )

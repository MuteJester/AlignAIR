from typing import Union

import numpy as np
from GenAIRR.dataconfig import DataConfig

from AlignAIR.Data import MultiDataConfigContainer
from AlignAIR.Step.Step import Step


class SegmentCorrectionStep(Step):
    def __init__(self,name):
        super().__init__(name)

    def calculate_pad_size(self,sequence, max_length=576):
        """
        Calculates the size of padding applied to each side of the sequence
        to achieve the specified maximum length.

        Args:
            sequence:
            sequence_length: The length of the original sequence before padding.
            max_length: The maximum length to which the sequence is padded.

        Returns:
            The size of the padding applied to the start of the sequence.
            If the total padding is odd, one additional unit of padding is applied to the end.
        """

        total_padding = max_length - len(sequence)
        pad_size = total_padding // 2

        return pad_size

    def correct_segments_for_paddings(self,sequences, dataconfig:Union[DataConfig, MultiDataConfigContainer], v_start, v_end, d_start, d_end, j_start, j_end):

        if isinstance(dataconfig, MultiDataConfigContainer):
            self.has_d = dataconfig.has_at_least_one_d()
        else:
            self.has_d = dataconfig.metadata.has_d

        paddings = np.array([self.calculate_pad_size(i) for i in sequences], dtype=np.int32)
        seq_lengths = np.array([len(i) for i in sequences], dtype=np.int32)

        def _sanitize_bounds(raw_start, raw_end):
            # Remove padding; use [start:end) with end-exclusive semantics
            s_raw = np.squeeze(raw_start)
            e_raw = np.squeeze(raw_end)
            # If logits argmax path used, these are already ints; otherwise floats
            s = np.floor(s_raw - paddings).astype(np.int32)
            e = np.floor(e_raw - paddings).astype(np.int32)
            # Clamp to valid range: start in [0, L-1], end in [1, L]
            s = np.clip(s, 0, seq_lengths - 1)
            e = np.clip(e, 1, seq_lengths)
            # Ensure non-empty end-exclusive interval
            e = np.maximum(e, s + 1)
            return s, e

        v_start, v_end = _sanitize_bounds(v_start, v_end)
        j_start, j_end = _sanitize_bounds(j_start, j_end)

        if self.has_d and d_start is not None and d_end is not None:
            d_start_arr = np.asarray(d_start)
            d_end_arr = np.asarray(d_end)
            d_start, d_end = _sanitize_bounds(d_start_arr, d_end_arr)
        else:
            d_start, d_end = None, None

        # Optional monotonic repair: enforce V ≤ D ≤ J ordering where applicable
        if self.has_d and d_start is not None and d_end is not None:
            # Ensure v_end ≤ d_start ≤ d_end ≤ j_start when possible
            d_start = np.maximum(d_start, v_end)
            d_end = np.maximum(d_end, d_start + 1)
            j_start = np.maximum(j_start, d_end)
            j_end = np.maximum(j_end, j_start + 1)
        else:
            # Enforce V then J ordering
            j_start = np.maximum(j_start, v_end)
            j_end = np.maximum(j_end, j_start + 1)

        cleaned_values = {'v_start':v_start, 'v_end':v_end,
                'd_start':d_start, 'd_end':d_end,
                'j_start':j_start, 'j_end':j_end}

        return cleaned_values

    def execute(self, predict_object):
        self.log("Correcting segments for paddings...")
        cleaned_data = predict_object.processed_predictions

        # add empty d_start and d_end if not present
        if 'd_start' not in cleaned_data:
            cleaned_data['d_start'] = None
        if 'd_end' not in cleaned_data:
            cleaned_data['d_end'] = None

        corrected_segments = self.correct_segments_for_paddings(
            predict_object.sequences, predict_object.dataconfig,
            cleaned_data['v_start'], cleaned_data['v_end'],
            cleaned_data['d_start'], cleaned_data['d_end'],
            cleaned_data['j_start'], cleaned_data['j_end']
        )
        for allele in ['v','d','j']:
            for segment in ['start','end']:
                predict_object.processed_predictions[f'{allele}_{segment}'] = corrected_segments[f'{allele}_{segment}']

        return predict_object

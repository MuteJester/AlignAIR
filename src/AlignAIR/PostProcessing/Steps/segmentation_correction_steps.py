import numpy as np

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

    def correct_segments_for_paddings(self,sequences, chain_type, v_start, v_end, d_start, d_end, j_start, j_end):
        paddings = np.array([self.calculate_pad_size(i) for i in sequences])

        v_start = np.round((v_start.squeeze() - paddings)).astype(int)
        v_end = np.round((v_end.squeeze() - paddings)).astype(int)

        j_start = np.round((j_start.squeeze() - paddings)).astype(int)
        j_end = np.round((j_end.squeeze() - paddings)).astype(int)

        if chain_type in ['heavy','tcrb']:
            d_start = np.round(np.vstack(d_start).squeeze() - paddings).astype(int)
            d_end = np.round(np.vstack(d_end).squeeze() - paddings).astype(int)
        else:
            d_start = None
            d_end = None

        return {'v_start':v_start, 'v_end':v_end,
                'd_start':d_start, 'd_end':d_end,
                'j_start':j_start, 'j_end':j_end}

    def execute(self, predict_object):
        self.log("Correcting segments for paddings...")
        cleaned_data = predict_object.processed_predictions
        corrected_segments = self.correct_segments_for_paddings(
            predict_object.sequences, predict_object.script_arguments.chain_type,
            cleaned_data['v_start'], cleaned_data['v_end'],
            cleaned_data['d_start'], cleaned_data['d_end'],
            cleaned_data['j_start'], cleaned_data['j_end']
        )
        for allele in ['v','d','j']:
            for segment in ['start','end']:
                predict_object.processed_predictions[f'{allele}_{segment}'] = corrected_segments[f'{allele}_{segment}']

        return predict_object

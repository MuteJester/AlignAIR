import numpy as np
from tqdm.auto import tqdm

class HeuristicReferenceMatcher:
    def __init__(self ,reference_alleles ,segment_threshold=0.2):
        self.reference_alleles = reference_alleles
    @staticmethod
    def hamming_distance(s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2)) + abs(len(s1 ) -len(s2))

    @staticmethod
    def calculate_pad_size(sequence, max_length=512):
        """
        Calculates the size of padding applied to each side of the sequence
        to achieve the specified maximum length.

        Args:
            sequence_length: The length of the original sequence before padding.
            max_length: The maximum length to which the sequence is padded.

        Returns:
            The size of the padding applied to the start of the sequence.
            If the total padding is odd, one additional unit of padding is applied to the end.
        """

        total_padding = max_length - len(sequence)
        pad_size = total_padding // 2

        return pad_size

    def AA_Score(self ,s1 ,s2):
        alignment_score = 0
        velocity = 0  # Initialize velocity
        acceleration = 0.05  # Define a constant for velocity adjustment

        last_match = None  # Track the last comparison result

        for c1, c2 in zip(s1, s2):
            is_match = c1 == c2
            if is_match:
                if last_match:
                    velocity += acceleration
                else:
                    velocity = acceleration
                score_change = -1 - velocity  # Negative score for a match
            else:
                if last_match == False:
                    velocity += acceleration
                else:
                    velocity = acceleration
                score_change = 1 + velocity  # Positive score for a mismatch

            alignment_score += score_change
            last_match = is_match

        return alignment_score

    def align_with_germline(self, short_segment, ref_seq, k=20, s=25):
        if len(short_segment) < 20:
            return -1,-1

        # take K bases from the start for the short segment and K bases from the end of the short segment
        L_seg =len(short_segment)
        L_ref = len(ref_seq)
        L_diff = L_ref - L_seg
        s = min(L_diff, s) + 1
        end_window = short_segment[-k:]

        # slide over the reference segment and look for the best poistion for the start and end
        min_difference = np.inf
        best_end_pos = L_ref
        for offset in range(0, s):  # +1 to include the last position
            ref_window = ref_seq[L_ref - (k + offset):(L_ref - offset)]
            difference = self.AA_Score(end_window, ref_window)
            if difference < min_difference:
                min_difference = difference
                best_end_pos = L_ref - offset
                if min_difference == 0:
                    break

        # Start window search refined
        min_difference = np.inf
        start_window = short_segment[:k]
        end_based_start = best_end_pos - L_seg
        best_start_pos = end_based_start
        # start_history = dict()
        # Adjust the search range based on potential indels
        start_search_range = min(9, L_diff)
        for offset in range(-start_search_range - 1, start_search_range + 1):
            current_start = max(0, end_based_start + offset)
            current_end = min(current_start + k, L_ref)
            ref_window = ref_seq[current_start:current_end]

            # Ensure the comparison is valid
            if len(ref_window) != len(start_window):
                continue  # Or use a different metric for unequal lengths

            difference = self.AA_Score(start_window, ref_window) + abs(offset)
            # start_history[current_start]=difference

            if difference < min_difference:
                min_difference = difference
                best_start_pos = current_start
                if difference == 0:
                    break



        return best_start_pos, best_end_pos

    def match(self, sequences, starts, ends, alleles,k=15,s=30):
        results = []

        # iterate over each sequence with its respective start and end positions as well as predicted allele
        for sequence, start, end, allele in tqdm(zip(sequences, starts, ends, alleles), total=len(starts)):
            # extract the portion of the sequence that based on the model predicted start and end positions
            segmented_sequence = sequence[start:end]
            # extract the reference allele
            reference_sequence = self.reference_alleles[allele]
            # calculate the reference and the germeline allele lengths
            segment_length = end - start
            reference_length = len(reference_sequence)


            match_found = False
            # Case 1: Exact Length Match
            if segment_length == reference_length:
                # if both extracted segment and reference are the same length we naively return the reference bounds
                results.append({'start_in_seq': start, 'end_in_seq': end,
                                'start_in_ref': 0, 'end_in_ref': reference_length})


            # Case 2: Extracted Segment is Longer than Reference
            elif segment_length > reference_length:
                # In this case one of two things could have caused this, either the AlignAIRR made a bad segmentation
                # Taking more bases than needed at the start or at the end, OR there were insertions in the sequence
                ref_start, ref_end = self.align_with_germline(segmented_sequence, reference_sequence, k=k, s=s)
                results.append({'start_in_seq': start, 'end_in_seq': end,
                                'start_in_ref': ref_start, 'end_in_ref': ref_end})

            # Case 3: Extracted Segment is Shorter than Reference
            elif segment_length < reference_length:
                # In this case one of two things could have caused this, either the AlignAIRR made a bad segmentation
                # Taking less bases than needed at the start or at the end, OR there were deletions in the sequence

                ref_start, ref_end = self.align_with_germline(segmented_sequence, reference_sequence, k=k, s=s)

                if ref_start is not None:
                    results.append({'start_in_seq': start, 'end_in_seq': end,
                                    'start_in_ref': ref_start, 'end_in_ref': ref_end})
                else:

                    # align sequences using local alignment
                    raise ValueError('Error')

        return results

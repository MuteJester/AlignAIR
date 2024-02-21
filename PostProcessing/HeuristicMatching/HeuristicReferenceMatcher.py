import numpy as np
from tqdm.auto import tqdm


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

class HeuristicReferenceMatcher:
    def __init__(self ,reference_alleles ,segment_threshold=0.2):
        self.reference_alleles = reference_alleles


    @staticmethod
    def apply_segment_thresholding(segments, t=0.3):
        mask = segments > t

        first_indices = np.full(mask.shape[0], -1, dtype=int)  # Fill with -1 to indicate "not found"
        last_indices = np.full(mask.shape[0], -1, dtype=int)

        for i, row in enumerate(mask):
            true_indices = np.where(row)[0]  # Find indices of True values in the row
            if true_indices.size > 0:
                first_indices[i] = true_indices[0]
                last_indices[i] = true_indices[-1]

        return first_indices, last_indices

    @staticmethod
    def hamming_distance(s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2)) + abs(len(s1 ) -len(s2))

    @staticmethod
    def hamming_similarity(s1, s2):
        return sum(c1 == c2 for c1, c2 in zip(s1, s2)) - abs(len(s1) - len(s2))

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

    def detect_indels(self, short_segment, ref_seq, k=20 ,s=25):
        if len(short_segment) < 20:
            indel_flag = False
            return indel_flag, -1, -1

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
        best_start_pos = None
        start_window = short_segment[:k]
        end_based_start = best_end_pos - L_seg
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

        # Indel flagging based on alignment length discrepancy
        best_match_length = best_end_pos - best_start_pos
        if best_match_length != L_seg:
            indel_flag = True
        else:
            indel_flag = False

        return indel_flag, best_start_pos, best_end_pos

    def find_best_end_match(self, short_end_segment, ref_seq, k):
        # Similar logic to find_best_start_match but starting from the end of ref_seq
        min_difference = np.inf
        best_pos = None
        L_ref = len(ref_seq)
        L_seg = len(short_end_segment)

        for offset in range(0, min(k + 1, L_ref - L_seg + 1)):
            ref_window = ref_seq[L_ref - L_seg - offset:L_ref - offset]
            difference = self.hamming_distance(short_end_segment, ref_window)
            if difference < min_difference:
                min_difference = difference
                best_pos = (L_ref - L_seg - offset, L_ref - offset)

        return best_pos

    def match(self, sequences, segments, alleles):
        starts, ends = self.apply_segment_thresholding(segments, t=0.2)
        padding_sizes = np.array([calculate_pad_size(seq) for seq in sequences])
        starts -= padding_sizes
        ends -= padding_sizes
        # correct for including some part of the mask by mistake
        starts[starts < 0] = 0
        ends[ends < 0] = 0

        results = []
        k = 15
        s = 30

        for sequence, start, end, allele in tqdm(zip(sequences, starts, ends, alleles), total=len(starts)):
            segmented_sequence = sequence[start:end]
            reference_sequence = self.reference_alleles[allele]
            segment_length = end - start
            reference_length = len(reference_sequence)
            match_found = False
            # Case 1: Exact Length Match
            if segment_length == reference_length:
                # if both extracted segment and reference are the same length we naively return the reference bounds
                results.append({'start_in_seq': start, 'end_in_seq': end,
                                'start_in_ref': 0, 'end_in_ref': reference_length, 'indel': False})


            # Case 2: Extracted Segment is Longer than Reference
            elif segment_length > reference_length:
                # In this case one of two things could have caused this, either the AlignAIRR made a bad segmentation
                # Taking more bases than needed at the start or at the end, OR there were insertions in the sequence
                best_start, best_end, _ = self.find_best_aligning_window(segmented_sequence, reference_sequence)
                results.append({'start_in_seq': start + best_start, 'end_in_seq': start + best_end,
                                'start_in_ref': 0, 'end_in_ref': len(reference_sequence),
                                'indel': False})

            # Case 3: Extracted Segment is Shorter than Reference
            elif segment_length < reference_length:
                # In this case one of two things could have caused this, either the AlignAIRR made a bad segmentation
                # Taking less bases than needed at the start or at the end, OR there were deletions in the sequence

                indel_flag, ref_start, ref_end = self.detect_indels(segmented_sequence, reference_sequence, k=k, s=s)

                if ref_start is not None:
                    results.append({'start_in_seq': start, 'end_in_seq': end,
                                    'start_in_ref': ref_start, 'end_in_ref': ref_end,
                                    'indel': indel_flag})
                else:

                    # align sequences using local alignment
                    raise ValueError('Error')

        return results

# # Example usage
# reference_alleles = {i.name: i.ungapped_seq.upper() for j in heavychain_config.v_alleles for i in
#                      heavychain_config.v_alleles[j]}
# sequences = ground_truth.sequence.to_list()
# mutation_ratios = mutation_rate.squeeze()
# top_references = predicted_alleles
#
# # Example usage
# mapper = HeuristicReferenceMatcher(reference_alleles)
# mappings = mapper.match(sequences=sequences, segments=v_segments, alleles=[i[0] for i in predicted_alleles])

import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Sequence, Tuple


class HeuristicReferenceMatcher:
    def __init__(self, reference_alleles: Dict[str, str], segment_threshold: float = 0.2):
        """
        Parameters
        ----------
        reference_alleles : mapping from allele‑name → uppercase ungapped sequence.
        segment_threshold : currently unused; reserved for future heuristics.
        """
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

    @staticmethod
    def _fast_tail_head_check(seg, ref, k=10, max_mm=3):
        """Return True iff first/last k‑mers of `seg` match those of `ref`
           with at most `max_mm` mismatches."""
        head_mismatch = sum(c1 != c2 for c1, c2 in zip(seg[:k], ref[:k]))
        tail_mismatch = sum(c1 != c2 for c1, c2 in zip(seg[-k:], ref[-k:]))
        return head_mismatch <= max_mm and tail_mismatch <= max_mm


    def _clip_overhang_no_indel(self, seq, start, end, ref_len):
        """
        If the extracted segment is longer than the reference and no indels are
        expected, trim equally from both sides so that len(seg)==len(ref).
        Returns new (start, end) tuple.
        """
        seg_len = end - start
        if seg_len <= ref_len:
            return start, end  # nothing to do

        excess = seg_len - ref_len
        start += excess // 2
        end -= excess - excess // 2
        return start, end

    def _affine_alignment_cost(self, s1: str, s2: str) -> float:
        score, vel = 0.0, 0.0
        accel = 0.05
        last_match = None

        for m in (c1 == c2 for c1, c2 in zip(s1, s2)):
            if m:
                vel = vel + accel if last_match else accel
                score -= 1 + vel
            else:
                vel = vel + accel if last_match is False else accel
                score += 1 + vel
            last_match = m
        return score

    @staticmethod
    def _is_pure_overhang(seg_len: int, ref_len: int, indels: int) -> bool:
        """True if extra length cannot be explained by biological indels."""
        return seg_len > ref_len or (seg_len == ref_len and indels == 0)

    def align_with_germline(self, short_segment, ref_seq,indel_count, k=20, s=25):
        # if len(short_segment) < k: # rethink this
        #     return -1,-1


        # take K bases from the start for the short segment and K bases from the end of the short segment
        L_seg =len(short_segment) # length of the sequence segment
        L_ref = len(ref_seq) # length of the reference sequence
        L_diff = abs(L_ref - L_seg) # length difference between the reference and the segment
        s = min(L_diff, s) + 1 # s is the search range
        end_window = short_segment[-k:] # take the last k bases from the short segment

        # slide over the reference segment and look for the best poistion for the start and end
        min_difference = np.inf
        best_end_pos = L_ref # we start by setting the best end position to the end of the reference sequence

        for offset in range(0, s):
            ref_window = ref_seq[L_ref - (k + offset):(L_ref - offset)]
            difference = self._affine_alignment_cost(end_window, ref_window)
            if difference < min_difference:
                min_difference = difference
                best_end_pos = L_ref - offset
                if min_difference == 0:
                    break

        # Start window search refined
        start_window = short_segment[:k] # take the first k bases from the short segment

        end_based_start = max(0,best_end_pos - L_seg) #set the start position based on the best end position
        best_start_pos = end_based_start
        min_difference = self._affine_alignment_cost(start_window, ref_seq[best_start_pos:best_start_pos+k])

        # start_history = dict()
        # Adjust the search range based on potential indels
        # Ensure indel_count is a plain Python int (handles numpy types/0-D arrays)
        try:
            indel_scalar = int(np.round(float(np.asarray(indel_count).squeeze())))
        except Exception:
            indel_scalar = int(indel_count)  # fallback
        indel_scalar = max(0, indel_scalar)

        if indel_scalar > 0:
            start_search_range = int(min(indel_scalar, int(L_diff)))
            search_iterator = range(-start_search_range - 1, start_search_range + 1)
        else:
            search_iterator = range(-1, 1)



        for offset in search_iterator:
            current_start = max(0, end_based_start + offset)
            current_end = min(current_start + k, L_ref)
            ref_window = ref_seq[current_start:current_end]

            # Ensure the comparison is valid
            if len(ref_window) != len(start_window):
                continue  # Or use a different metric for unequal lengths

            difference = self._affine_alignment_cost(start_window, ref_window) + abs(offset)
            # start_history[current_start]=difference

            if difference < min_difference:
                min_difference = difference
                best_start_pos = current_start
                if difference == 0:
                    break



        return best_start_pos, best_end_pos

    def match(self, sequences, starts, ends, alleles,
              indel_counts, k=15, s=30, _gene=None):

        results = []
        desc = f'Matching {_gene.upper()} Germlines'

        for seq, start, end, allele, indels in tqdm(
                                                    zip(sequences, starts, ends, alleles, indel_counts),
                                                    total=len(starts), desc=desc
                                                    ):

            ref_seq = self.reference_alleles[allele]
            ref_len = len(ref_seq)

            # ──────────  hard guard for overhang w/ zero indels ──────────
            if indels == 0 and (end - start) > ref_len:
                start, end = self._clip_overhang_no_indel(seq, start, end, ref_len)
            # -----------------------------------------------------------------

            seg = seq[start:end]
            seg_len = len(seg)

            # ---------- Case 1 : same length ---------------------------------
            if seg_len == ref_len:
                if indels == 0 and self._fast_tail_head_check(seg, ref_seq, k=10):
                    # confident quick exit
                    results.append(dict(start_in_seq=start, end_in_seq=end,
                                        start_in_ref=0, end_in_ref=ref_len))
                    continue
                # otherwise fall through to align_with_germline
            # -----------------------------------------------------------------

            orig_seg_len = end - start


            # ---------- Cases 2 & 3 : length differs -------------------------
            ref_start, ref_end = self.align_with_germline(
                seg, ref_seq, indels, k=k, s=s
            )

            is_overhang = self._is_pure_overhang(orig_seg_len, ref_len, indels)


            if is_overhang and (ref_start > 0 or ref_end < ref_len):
                start += ref_start
                end -= (ref_len - ref_end)
                start = max(start, 0)
                end = min(end, len(seq))
                ref_start, ref_end = 0, ref_len


            results.append(dict(start_in_seq=start, end_in_seq=end,
                                start_in_ref=ref_start, end_in_ref=ref_end))
        return results


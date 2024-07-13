from AlignAIR.PostProcessing import HeuristicReferenceMatcher
from AlignAIR.Step.Step import Step


class AlleleAlignmentStep(Step):

    def align_with_germline(self,segments, threshold_objects, predicted_alleles, sequences):
        germline_alignmnets = {}

        for _gene in segments:
            reference_alleles = threshold_objects[_gene].reference_map[_gene]
            if _gene == 'd':
                reference_alleles['Short-D'] = ''

            starts, ends = segments[_gene]
            mapper = HeuristicReferenceMatcher(reference_alleles)
            mappings = mapper.match(sequences=sequences, starts=starts, ends=ends,
                                    alleles=[i[0] for i in predicted_alleles[_gene]], _gene=_gene)

            germline_alignmnets[_gene] = mappings

        return germline_alignmnets
    def execute(self, predict_object):
        self.log("Aligning with germline...")
        allele_info = predict_object.results['allele_info']
        corrected_segments = predict_object.results['corrected_segments']
        segments = {'v': [corrected_segments['v_start'],corrected_segments['v_end']],
                    'j': [corrected_segments['j_start'],corrected_segments['j_end']]}
        if predict_object.chain_type == 'heavy':
            segments['d'] = [corrected_segments['d_start'],corrected_segments['d_end']]

        predict_object.results['germline_alignments'] = self.align_with_germline(
            segments, allele_info[-1], allele_info[0], predict_object.sequences)
        return predict_object

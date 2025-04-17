from AlignAIR.PostProcessing import HeuristicReferenceMatcher
from AlignAIR.Step.Step import Step


class AlleleAlignmentStep(Step):

    def __init__(self, name):
        super().__init__(name)

    def align_with_germline(self, segments, threshold_objects, predicted_alleles, sequences,indel_counts):
        germline_alignmnets = {}

        for _gene in segments:
            reference_alleles = threshold_objects[_gene].reference_map[_gene]
            if _gene == 'd':
                reference_alleles['Short-D'] = ''

            starts, ends = segments[_gene]
            mapper = HeuristicReferenceMatcher(reference_alleles)
            mappings = mapper.match(sequences=sequences, starts=starts, ends=ends,
                                    alleles=[i[0] for i in predicted_alleles[_gene]], _gene=_gene,indel_counts=indel_counts)

            germline_alignmnets[_gene] = mappings

        return germline_alignmnets

    def execute(self, predict_object):
        self.log("Aligning with germline alleles...")

        processed_predictions = predict_object.processed_predictions
        segments = {}
        iterator = ['v','j']
        if predict_object.data_config_library.mounted == 'heavy':
            iterator.append('d')

        for gene in iterator:
            segments[f'{gene}'] = (processed_predictions[f'{gene}_start'], processed_predictions[f'{gene}_end'])

        predict_object.germline_alignments = self.align_with_germline(
            segments=segments,
            threshold_objects=predict_object.threshold_extractor_instances,
            predicted_alleles=predict_object.selected_allele_calls,
            sequences=predict_object.sequences,
            indel_counts= processed_predictions['indel_count']
        )

        return predict_object

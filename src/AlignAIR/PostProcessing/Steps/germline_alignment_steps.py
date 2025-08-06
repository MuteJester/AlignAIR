from GenAIRR.dataconfig import DataConfig

from AlignAIR.Data import MultiDataConfigContainer
from AlignAIR.PostProcessing import HeuristicReferenceMatcher
from AlignAIR.Step.Step import Step


class AlleleAlignmentStep(Step):

    def __init__(self, name):
        super().__init__(name)

    def align_with_germline(self, segments,dataconfig:DataConfig, predicted_alleles, sequences,indel_counts):
        germline_alignmnets = {}
        reference_map = {}
        reference_map['v'] = {i.name:i.ungapped_seq.upper() for i in dataconfig.allele_list('v')}
        reference_map['j'] = {i.name:i.ungapped_seq.upper() for i in dataconfig.allele_list('j')}
        if self.has_d:
            reference_map['d'] = {i.name:i.ungapped_seq.upper() for i in dataconfig.allele_list('d')}
            reference_map['d']['Short-D'] = ''

        for _gene in segments:
            reference_alleles = reference_map[_gene]
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

        if isinstance(predict_object.dataconfig,DataConfig):
            self.has_d = predict_object.dataconfig.metadata.has_d
        elif isinstance(predict_object.dataconfig,MultiDataConfigContainer):
            self.has_d = predict_object.dataconfig.has_at_least_one_d()
        else:
            raise ValueError("dataconfig should be either a DataConfig or MultiDataConfigContainer")


        if self.has_d:
            iterator.append('d')

        for gene in iterator:
            segments[f'{gene}'] = (processed_predictions[f'{gene}_start'], processed_predictions[f'{gene}_end'])

        indel_counts = processed_predictions['indel_count'].round().astype(int)

        predict_object.germline_alignments = self.align_with_germline(
            segments=segments,
            dataconfig=predict_object.dataconfig,
            predicted_alleles=predict_object.selected_allele_calls,
            sequences=predict_object.sequences,
            indel_counts=indel_counts

        )

        return predict_object

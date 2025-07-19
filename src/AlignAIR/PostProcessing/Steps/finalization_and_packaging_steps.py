import pandas as pd

from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Step.Step import Step


class FinalizationStep(Step):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, predict_object: PredictObject):
        self.log("Finalizing results and saving to CSV...")
        cleaned_data = predict_object.processed_predictions
        germline_alignments = predict_object.germline_alignments

        sequences = predict_object.sequences
        save_path = predict_object.script_arguments.save_path
        file_name = predict_object.file_info.file_name

        # Compile results into a DataFrame
        final_csv = pd.DataFrame({
            'sequence': sequences,
            'v_call': [','.join(i) for i in predict_object.selected_allele_calls['v']],
            'j_call': [','.join(i) for i in predict_object.selected_allele_calls['j']],
            'v_sequence_start': [i['start_in_seq'] for i in predict_object.germline_alignments['v']],
            'v_sequence_end': [i['end_in_seq'] for i in predict_object.germline_alignments['v']],
            'j_sequence_start': [i['start_in_seq'] for i in predict_object.germline_alignments['j']],
            'j_sequence_end': [i['end_in_seq'] for i in predict_object.germline_alignments['j']],
            'v_germline_start': [max(0, i['start_in_ref']) for i in predict_object.germline_alignments['v']],
            'v_germline_end': [i['end_in_ref'] for i in predict_object.germline_alignments['v']],
            'j_germline_start': [max(0, i['start_in_ref']) for i in predict_object.germline_alignments['j']],
            'j_germline_end': [i['end_in_ref'] for i in predict_object.germline_alignments['j']],
            'v_likelihoods': predict_object.likelihoods_of_selected_alleles['v'],
            'j_likelihoods': predict_object.likelihoods_of_selected_alleles['j'],
            'mutation_rate': predict_object.processed_predictions['mutation_rate'],
            'ar_indels': predict_object.processed_predictions['indel_count'],
            'ar_productive': predict_object.processed_predictions['productive'],
        })

        if predict_object.dataconfig.metadata.has_d:
            final_csv['d_sequence_start'] = [i['start_in_seq'] for i in predict_object.germline_alignments['d']]
            final_csv['d_sequence_end'] = [i['end_in_seq'] for i in predict_object.germline_alignments['d']]
            final_csv['d_germline_start'] = [abs(i['start_in_ref']) for i in predict_object.germline_alignments['d']]
            final_csv['d_germline_end'] = [i['end_in_ref'] for i in predict_object.germline_alignments['d']]
            final_csv['d_call'] = [','.join(i) for i in predict_object.selected_allele_calls['d']]
            final_csv['d_likelihoods'] = predict_object.likelihoods_of_selected_alleles['d']
            final_csv['type'] = predict_object.dataconfig.metadata.chain_type
        else:
            final_csv['type'] = ['kappa' if i == 1 else 'lambda' for i in predict_object.processed_predictions['type_'].astype(int).squeeze()]

        # Save to CSV
        final_csv_path = f"{save_path}{file_name}_alignairr_results.csv"
        final_csv.to_csv(final_csv_path, index=False)
        self.log(f"Results saved successfully at {final_csv_path}")

        return predict_object

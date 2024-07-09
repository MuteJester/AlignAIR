import pandas as pd

from AlignAIR.Step.Step import Step


class FinalizationStep(Step):
    def __init__(self, name, logger=None):
        super().__init__(name, logger)

    def execute(self, predict_object):
        self.log("Finalizing results and saving to CSV...")
        cleaned_data = predict_object.results['cleaned_data']
        alignments = predict_object.results['germline_alignments']
        predict_object.final_results = {
            'predicted_alleles': predict_object.results['allele_info'][0],
            'germline_alignments': alignments,
            'predicted_allele_likelihoods': predict_object.results['allele_info'][1],
            'mutation_rate': cleaned_data['mutation_rate'],
            'productive': cleaned_data['productive'],
            'indel_count': cleaned_data['indel_count']
        }
        if predict_object.chain_type == 'light':
            predict_object.final_results['type_'] = cleaned_data['type_']


        results = predict_object.final_results
        sequences = predict_object.sequences
        chain_type = predict_object.script_arguments.chain_type
        save_path = predict_object.script_arguments.save_path
        file_name = predict_object.file_name  # Ensure this is part of predict_object.config or similar

        # Compile results into a DataFrame
        final_csv = pd.DataFrame({
            'sequence': sequences,
            'v_call': [','.join(i) for i in results['predicted_alleles']['v']],
            'j_call': [','.join(i) for i in results['predicted_alleles']['j']],
            'v_sequence_start': [i['start_in_seq'] for i in results['germline_alignments']['v']],
            'v_sequence_end': [i['end_in_seq'] for i in results['germline_alignments']['v']],
            'j_sequence_start': [i['start_in_seq'] for i in results['germline_alignments']['j']],
            'j_sequence_end': [i['end_in_seq'] for i in results['germline_alignments']['j']],
            'v_germline_start': [max(0, i['start_in_ref']) for i in results['germline_alignments']['v']],
            'v_germline_end': [i['end_in_ref'] for i in results['germline_alignments']['v']],
            'j_germline_start': [max(0, i['start_in_ref']) for i in results['germline_alignments']['j']],
            'j_germline_end': [i['end_in_ref'] for i in results['germline_alignments']['j']],
            'v_likelihoods': results['predicted_allele_likelihoods']['v'],
            'j_likelihoods': results['predicted_allele_likelihoods']['j'],
            'mutation_rate': results['mutation_rate'],
            'ar_indels': results['indel_count'],
            'ar_productive': results['productive'],
        })

        if chain_type == 'heavy':
            final_csv['d_sequence_start'] = [i['start_in_seq'] for i in results['germline_alignments']['d']]
            final_csv['d_sequence_end'] = [i['end_in_seq'] for i in results['germline_alignments']['d']]
            final_csv['d_germline_start'] = [abs(i['start_in_ref']) for i in results['germline_alignments']['d']]
            final_csv['d_germline_end'] = [i['end_in_ref'] for i in results['germline_alignments']['d']]
            final_csv['d_call'] = [','.join(i) for i in results['predicted_alleles']['d']]
            final_csv['type'] = 'heavy'
        else:
            final_csv['type'] = ['kappa' if i == 1 else 'lambda' for i in results['type_'].astype(int).squeeze()]

        # Save to CSV
        final_csv_path = f"{save_path}{file_name}_alignairr_results.csv"
        final_csv.to_csv(final_csv_path, index=False)
        self.log(f"Results saved successfully at {final_csv_path}")

        return predict_object

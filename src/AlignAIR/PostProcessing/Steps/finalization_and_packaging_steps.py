import os
from pathlib import Path

import pandas as pd
from GenAIRR.dataconfig import DataConfig

from AlignAIR.Data import MultiDataConfigContainer
from AlignAIR.Data.encoders import ChainTypeOneHotEncoder
from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Step.Step import Step


class FinalizationStep(Step):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, predict_object: PredictObject):
        self.log("Finalizing results and saving to CSV...")
        cleaned_data = predict_object.processed_predictions
        germline_alignments = predict_object.germline_alignments

        if isinstance(predict_object.dataconfig, DataConfig):
            self.has_d = predict_object.dataconfig.metadata.has_d
        elif isinstance(predict_object.dataconfig, MultiDataConfigContainer):
            self.has_d = predict_object.dataconfig.has_at_least_one_d()
        else:
            raise ValueError("dataconfig should be either a DataConfig or MultiDataConfigContainer")

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
            'indels': predict_object.processed_predictions['indel_count'],
            'productive': predict_object.processed_predictions['productive'],
        })

        if self.has_d:
            final_csv['d_sequence_start'] = [i['start_in_seq'] for i in predict_object.germline_alignments['d']]
            final_csv['d_sequence_end'] = [i['end_in_seq'] for i in predict_object.germline_alignments['d']]
            final_csv['d_germline_start'] = [abs(i['start_in_ref']) for i in predict_object.germline_alignments['d']]
            final_csv['d_germline_end'] = [i['end_in_ref'] for i in predict_object.germline_alignments['d']]
            final_csv['d_call'] = [','.join(i) for i in predict_object.selected_allele_calls['d']]
            final_csv['d_likelihoods'] = predict_object.likelihoods_of_selected_alleles['d']
            final_csv['chain_type'] = predict_object.dataconfig.metadata.chain_type

        if isinstance(predict_object.dataconfig, MultiDataConfigContainer):
          chaintype_ohe = ChainTypeOneHotEncoder(chain_types=predict_object.dataconfig.chain_types())
          decoded_types = chaintype_ohe.decode(predict_object.processed_predictions['type_'])

          final_csv['chain_type'] = decoded_types

        path_obj = Path(save_path)
       # Check if the provided path ends with '.csv'
        if path_obj.suffix.lower() == '.csv':
            # User provided a full file path, use it directly
            final_csv_path = path_obj
            # Ensure the directory for the custom file path exists
            # final_csv_path.parent gives the directory part: /this/is/a/custom/file/
            os.makedirs(final_csv_path.parent, exist_ok=True)
        else:
            # User provided a directory, so construct the filename as before
            # Ensure the directory exists
            os.makedirs(path_obj, exist_ok=True)
            # Use the / operator to safely join the path and the new filename
            file_to_save = f"{file_name}_alignairr_results.csv"
            final_csv_path = path_obj / file_to_save

        final_csv.to_csv(final_csv_path, index=False)

        self.log(f"Results saved successfully at {final_csv_path}")

        return predict_object

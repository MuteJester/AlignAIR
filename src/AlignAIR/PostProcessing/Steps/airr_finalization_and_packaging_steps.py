from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Step.Step import Step
from AlignAIR.PostProcessing import AIRRFormatManager
from AlignAIR.PostProcessing import TranslateToIMGT

class AIRRFinalizationStep(Step):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, predict_object: PredictObject):
        self.log("Finalizing AIRR results and saving to TSV...")
        
        save_path = predict_object.script_arguments.save_path
        file_name = predict_object.file_info.file_name
        
        airr_formatter = AIRRFormatManager(predict_object)
        final_tsv = airr_formatter.build_dataframe()
        
        if not predict_object.script_arguments.translate_to_asc:
            self.log("Translating allele names...")
            translator = TranslateToIMGT(predict_object.data_config_library.packaged_config())
            predict_object.selected_allele_calls['v'] = [
                [translator.translate(allele) for allele in alleles]
                for alleles in predict_object.selected_allele_calls['v']
            ]
            final_tsv['v_call'] = [','.join(alleles) for alleles in predict_object.selected_allele_calls['v']]

        # Save to TSV
        final_path = f"{save_path}{file_name}_alignair_results.tsv"
        final_tsv.to_csv(final_path, sep='\t', index=False)
        self.log(f"Results saved successfully at {final_path}")

        return predict_object

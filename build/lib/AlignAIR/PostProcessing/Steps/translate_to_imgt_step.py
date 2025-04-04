from AlignAIR.PostProcessing import TranslateToIMGT
from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Step.Step import Step


class TranslationStep(Step):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, predict_object:PredictObject):
        self.log("Translating allele names...")
        # Check if translat ion is needed
        if not predict_object.script_arguments.translate_to_asc:
            translator = TranslateToIMGT(predict_object.data_config_library.packaged_config())
            # Assuming 'v' allele needs translation and it's stored in a specific key
            predict_object.selected_allele_calls['v'] = [
                [translator.translate(j) for j in i] for i in predict_object.selected_allele_calls['v']]
        return predict_object

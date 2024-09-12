from AlignAIR.PostProcessing import TranslateToIMGT
from AlignAIR.Step.Step import Step


class TranslationStep(Step):
    def __init__(self, name, logger=None):
        super().__init__(name, logger)

    def execute(self, predict_object):
        self.log("Translating allele names...")
        # Check if translat ion is needed
        if not predict_object.script_arguments.translate_to_asc:
            translator = TranslateToIMGT(predict_object.data_config)
            # Assuming 'v' allele needs translation and it's stored in a specific key
            predict_object.results['allele_info'][0]['v'] = [
                [translator.translate(j) for j in i] for i in predict_object.results['allele_info'][0]['v']]
        return predict_object

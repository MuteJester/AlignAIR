import numpy as np
from AlignAIR.Step.Step import Step


class CleanAndArrangeStep(Step):

    def __init__(self,name):
        super().__init__(name)



    def clean_and_arrange_predictions(self, predictions, dataconfig):
        def extract_values(key):
            return [i[key] for i in predictions]

        mutation_rate = np.squeeze(np.vstack(extract_values('mutation_rate')))
        indel_count = np.squeeze(np.vstack(extract_values('indel_count')))
        productive = np.squeeze(np.vstack(extract_values('productive')) > 0.5)

        v_allele = np.vstack(extract_values('v_allele'))
        j_allele = np.vstack(extract_values('j_allele'))
        v_start = np.vstack(extract_values('v_start'))
        v_end = np.vstack(extract_values('v_end'))
        j_start = np.vstack(extract_values('j_start'))
        j_end = np.vstack(extract_values('j_end'))

        if dataconfig.metadata.has_d:
            d_allele = np.vstack(extract_values('d_allele'))
            d_start = np.vstack(extract_values('d_start'))
            d_end = np.vstack(extract_values('d_end'))
        else:
            d_allele = None
            d_start = None
            d_end = None

        type_ = None


        output = {
            'v_allele': v_allele,
            'j_allele': j_allele,
            'v_start': v_start,
            'v_end': v_end,
            'j_start': j_start,
            'j_end': j_end,
            'mutation_rate': mutation_rate,
            'indel_count': indel_count,
            'productive': productive,
            #'type_': type_ if chain_type == 'light' else None
        }
        if dataconfig.metadata.has_d:
            output[ 'd_allele'] =  d_allele
            output['d_start'] = d_start
            output['d_end'] = d_end

        return output
    def execute(self, predict_object):
        self.log("Cleaning and arranging predictions...")
        predict_object.processed_predictions = self.clean_and_arrange_predictions(
            predict_object.raw_predictions,
            predict_object.dataconfig
        )

        return predict_object

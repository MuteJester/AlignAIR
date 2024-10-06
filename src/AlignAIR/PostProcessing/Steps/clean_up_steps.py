import numpy as np

from AlignAIR.Step.Step import Step


class CleanAndArrangeStep(Step):

    def clean_and_arrange_predictions(self, predictions, chain_type):
        mutation_rate, v_allele, d_allele, j_allele = [], [], [], []
        v_start, v_end = [], []
        d_start, d_end = [], []
        j_start, j_end = [], []
        indel_count = []
        type_ = []
        productive = []
        for i in predictions:
            mutation_rate.append(i['mutation_rate'])
            v_allele.append(i['v_allele'])
            j_allele.append(i['j_allele'])
            indel_count.append(i['indel_count'])
            productive.append(i['productive'])

            v_start.append(i['v_start'])
            v_end.append(i['v_end'])
            j_start.append(i['j_start'])
            j_end.append(i['j_end'])

            if chain_type == 'light':
                type_.append(i['type'])
            else:
                d_start.append(i['d_start'])
                d_allele.append(i['d_allele'])
                d_end.append(i['d_end'])

        mutation_rate = np.vstack(mutation_rate)
        indel_count = np.vstack(indel_count)
        productive = np.vstack(productive) > 0.5

        productive = productive.squeeze()
        indel_count = indel_count.squeeze()
        mutation_rate = mutation_rate.squeeze()

        v_allele = np.vstack(v_allele)
        if chain_type == 'heavy':
            d_allele = np.vstack(d_allele)
        j_allele = np.vstack(j_allele)

        v_start = np.vstack(v_start)
        v_end = np.vstack(v_end)

        j_start = np.vstack(j_start)
        j_end = np.vstack(j_end)

        if chain_type == 'light':
            type_ = np.vstack(type_)
        else:
            d_start = np.vstack(d_start)
            d_end = np.vstack(d_end)
            d_allele = np.vstack(d_allele)

        output = {'v_allele': v_allele, 'd_allele': d_allele,
                  'j_allele': j_allele, 'v_start': v_start, 'v_end': v_end,
                  'd_start': d_start, 'd_end': d_end,
                  'j_start': j_start, 'j_end': j_end,
                  'mutation_rate': mutation_rate, 'indel_count': indel_count,
                  'productive': productive, 'type_': type_
                  }

        return output
    def execute(self, predict_object):
        self.log("Cleaning and arranging predictions...")
        predict_object.results['cleaned_data'] = self.clean_and_arrange_predictions(
            predict_object.results['predictions'], predict_object.script_arguments.chain_type)
        return predict_object

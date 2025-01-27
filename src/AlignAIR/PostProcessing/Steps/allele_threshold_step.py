from AlignAIR.PostProcessing.AlleleSelector import CappedDynamicConfidenceThreshold, MaxLikelihoodPercentageThreshold
from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Step.Step import Step
from AlignAIR.Utilities.step_utilities import DataConfigLibrary


class ConfidenceMethodThresholdApplicationStep(Step):

    def extract_likelihoods_and_labels_from_calls(self, args, alleles, threshold, caps, config):
        predicted_alleles = {}
        predicted_allele_likelihoods = {}
        threshold_objects = {}
        chain_type = args.chain_type

        for _gene in alleles:
            if chain_type == 'heavy':
                extractor = CappedDynamicConfidenceThreshold(heavy_dataconfig=config['heavy'])
            else:
                extractor = CappedDynamicConfidenceThreshold(kappa_dataconfig=config['kappa'],
                                                             lambda_dataconfig=config['lambda'])

            threshold_objects[_gene] = extractor
            selected_alleles = extractor.get_alleles(alleles[_gene], confidence=threshold[_gene],
                                                     cap=caps[_gene], allele=_gene, verbose=True)

            predicted_alleles[_gene] = [i[0] for i in selected_alleles]

            predicted_allele_likelihoods[_gene] = [i[1] for i in selected_alleles]

        return predicted_alleles, predicted_allele_likelihoods, threshold_objects

    def execute(self, predict_object):
        self.log("Applying dynamic confidence thresholds...")
        args = predict_object.script_arguments
        alleles = {'v': predict_object.results['cleaned_data']['v_allele'],
                   'j': predict_object.results['cleaned_data']['j_allele']}
        thresholds = {'v': args.v_allele_threshold, 'd': args.d_allele_threshold, 'j': args.j_allele_threshold}
        caps = {'v': args.v_cap, 'd': args.d_cap, 'j': args.j_cap}

        if args.chain_type == 'heavy':
            alleles['d'] = predict_object.results['cleaned_data']['d_allele']

        predict_object.results['allele_info'] = self.extract_likelihoods_and_labels_from_calls(
            predict_object.script_arguments, alleles, thresholds,
            caps, predict_object.data_config)
        return predict_object


class MaxLikelihoodPercentageThresholdApplicationStep(Step):

    def __init__(self, name):
        super().__init__(name)

    def extract_likelihoods_and_labels_from_calls(self, alleles, threshold, caps,
                                                  data_config_library: DataConfigLibrary):
        predicted_alleles = {}
        predicted_allele_likelihoods = {}
        threshold_objects = {}


        for _gene in alleles:
            if data_config_library.mounted == 'heavy':
                extractor = MaxLikelihoodPercentageThreshold(heavy_dataconfig=data_config_library.config())
            else:
                extractor = MaxLikelihoodPercentageThreshold(kappa_dataconfig=data_config_library.config('kappa'),
                                                             lambda_dataconfig=data_config_library.config('lambda'))

            threshold_objects[_gene] = extractor
            selected_alleles = extractor.get_alleles(alleles[_gene], percentage=threshold[_gene],
                                                     cap=caps[_gene], allele=_gene, verbose=True)

            predicted_alleles[_gene] = [i[0] for i in selected_alleles]

            predicted_allele_likelihoods[_gene] = [i[1] for i in selected_alleles]

        return predicted_alleles, predicted_allele_likelihoods, threshold_objects

    @staticmethod
    def get_thresholds(args):
        """extract v,d,j thresholds from args"""
        return {'v': args.v_allele_threshold, 'd': args.d_allele_threshold, 'j': args.j_allele_threshold}

    @staticmethod
    def get_caps(args):
        """extract v,d,j caps from args"""
        return {'v': args.v_cap, 'd': args.d_cap, 'j': args.j_cap}

    def execute(self, predict_object : PredictObject):
        self.log("Applying Max Likelihood thresholds...")
        args = predict_object.script_arguments

        alleles = {'v': predict_object.processed_predictions['v_allele'],
                   'j': predict_object.processed_predictions['j_allele'],
                   }
        if predict_object.data_config_library.mounted == 'heavy':
            alleles['d'] = predict_object.processed_predictions['d_allele']

        thresholds = self.get_thresholds(args)
        caps = self.get_caps(args)

        (predict_object.selected_allele_calls,
         predict_object.likelihoods_of_selected_alleles,
         predict_object.threshold_extractor_instances) \
            = self.extract_likelihoods_and_labels_from_calls(
                                                                alleles,
                                                                thresholds,
                                                                caps,
                                                                predict_object.data_config_library
                                                            )

        return predict_object

from typing import Dict, Union

import numpy as np
from GenAIRR.dataconfig import DataConfig

from AlignAIR.Data import MultiDataConfigContainer
from AlignAIR.PostProcessing.AlleleSelector import CappedDynamicConfidenceThreshold, MaxLikelihoodPercentageThreshold
from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Step.Step import Step


class ConfidenceMethodThresholdApplicationStep(Step):

    def extract_likelihoods_and_labels_from_calls(self,
                                                  args,
                                                  predicted_allele_likelihoods: Dict[str, np.ndarray],
                                                  likelihood_thresholds: Dict[str, float],
                                                  call_caps: Dict[str, int],
                                                  dataconfig: Union[DataConfig, MultiDataConfigContainer]):
        predicted_alleles = {}
        processed_predicted_allele_likelihoods = {}
        threshold_objects = {}
        extractor = CappedDynamicConfidenceThreshold(dataconfig=dataconfig)

        for _gene in likelihood_thresholds:
            threshold_objects[_gene] = extractor
            selected_alleles = extractor.get_alleles(predicted_allele_likelihoods[_gene],
                                                     confidence=likelihood_thresholds[_gene],
                                                     cap=call_caps[_gene],
                                                     allele=_gene,
                                                     verbose=True)

            predicted_alleles[_gene] = [i[0] for i in selected_alleles]

            processed_predicted_allele_likelihoods[_gene] = [i[1] for i in selected_alleles]

        return predicted_alleles, processed_predicted_allele_likelihoods, threshold_objects

    def execute(self, predict_object):
        self.log("Applying dynamic confidence thresholds...")
        args = predict_object.script_arguments
        alleles = {'v': predict_object.results['cleaned_data']['v_allele'],
                   'j': predict_object.results['cleaned_data']['j_allele']}
        thresholds = {'v': args.v_allele_threshold, 'd': args.d_allele_threshold, 'j': args.j_allele_threshold}
        caps = {'v': args.v_cap, 'd': args.d_cap, 'j': args.j_cap}

        if args.chain_type in ['heavy','tcrb']:
            alleles['d'] = predict_object.results['cleaned_data']['d_allele']

        predict_object.results['allele_info'] = self.extract_likelihoods_and_labels_from_calls(
            predict_object.script_arguments, alleles, thresholds,
            caps, predict_object.data_config)
        return predict_object


class MaxLikelihoodPercentageThresholdApplicationStep(Step):

    def __init__(self, name):
        super().__init__(name)

    def extract_likelihoods_and_labels_from_calls(self,
                                                  predicted_allele_likelihoods: Dict[str, np.ndarray],
                                                  likelihood_thresholds: Dict[str, float],
                                                  call_caps: Dict[str, int],
                                                  dataconfig: Union[DataConfig, MultiDataConfigContainer]):
        predicted_alleles = {}
        processed_predicted_allele_likelihoods = {}
        threshold_objects = {}

        extractor = MaxLikelihoodPercentageThreshold(dataconfig=dataconfig)

        for _gene in likelihood_thresholds:
            threshold_objects[_gene] = extractor
            selected_alleles = extractor.get_alleles(predicted_allele_likelihoods[_gene],
                                                     percentage=likelihood_thresholds[_gene],
                                                     cap=call_caps[_gene],
                                                     allele=_gene,
                                                     verbose=True)

            predicted_alleles[_gene] = [i[0] for i in selected_alleles]

            processed_predicted_allele_likelihoods[_gene] = [i[1] for i in selected_alleles]

        return predicted_alleles, processed_predicted_allele_likelihoods, threshold_objects


    def get_thresholds(self,args):
        """extract v,d,j thresholds from args"""
        ths =  {'v': args.v_allele_threshold,  'j': args.j_allele_threshold}
        if self.has_d:
            ths['d'] = args.d_allele_threshold

        return ths

    @staticmethod
    def get_caps(args):
        """extract v,d,j caps from args"""
        return {'v': args.v_cap, 'd': args.d_cap, 'j': args.j_cap}

    def execute(self, predict_object : PredictObject):
        self.log("Applying Max Likelihood thresholds...")
        args = predict_object.script_arguments

        # Lint-safe guard: ensure processed_predictions exists
        if not getattr(predict_object, 'processed_predictions', None):
            self.log("No processed_predictions available; skipping thresholding step.")
            return predict_object

        if isinstance(predict_object.dataconfig, DataConfig):
            self.has_d = predict_object.dataconfig.metadata.has_d
        elif isinstance(predict_object.dataconfig,MultiDataConfigContainer):
            self.has_d = predict_object.dataconfig.has_at_least_one_d()
        else:
            raise ValueError("dataconfig should be a DataConfig or MultiDataConfigContainer instance")

        alleles = {'v': predict_object.processed_predictions['v_allele'],
                   'j': predict_object.processed_predictions['j_allele'],
                   }
        if self.has_d:
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
                                                                predict_object.dataconfig
                                                            )

        # Diagnostics: summarize selected calls and their likelihoods
        try:
            import numpy as _np
            sels = predict_object.selected_allele_calls
            likes = predict_object.likelihoods_of_selected_alleles
            def _stat(v):
                try:
                    arr = _np.array(v, dtype=float)
                    if arr.size == 0:
                        return {'count': 0}
                    return {
                        'count': int(arr.size),
                        'mean': float(_np.nanmean(arr)),
                        'std': float(_np.nanstd(arr)),
                        'min': float(_np.nanmin(arr)),
                        'max': float(_np.nanmax(arr)),
                    }
                except Exception:
                    return {'count': len(v) if hasattr(v, '__len__') else 0}
            v_stats = _stat(likes.get('v', []))
            j_stats = _stat(likes.get('j', []))
            d_stats = _stat(likes.get('d', [])) if 'd' in likes else None
            self.log(f"Selected allele likelihoods — V: {v_stats}; J: {j_stats}; D: {d_stats}")
        except Exception as _sel_err:
            self.log(f"Selection diagnostics skipped due to error: {_sel_err}")

        return predict_object

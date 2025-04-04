import pandas as pd

from AlignAIR.Step.Step import Step
from GenAIRR.data import builtin_kappa_chain_data_config, builtin_lambda_chain_data_config, \
    builtin_heavy_chain_data_config
import pickle
class LoadGroundTruthStep(Step):
    def __init__(self, name, logger=None):
        super().__init__(name, logger)

    def execute(self, predict_object):
        predict_object.groundtruth_table = pd.read_csv(predict_object.script_arguments.sequences)
        return predict_object

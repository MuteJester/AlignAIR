from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Step.Step import Step
from AlignAIR.Utilities.step_utilities import DataConfigLibrary


class ConfigLoadStep(Step):

    def __init__(self, name):
        super().__init__(name)

    def process(self, predict_object: PredictObject):
        """
        Loads configuration based on the chain type from provided paths.

        Args:
            predict_object (PredictObject): The object that holds the chain_type and config_paths.

        Returns:
            PredictObject: Updated with loaded configuration.
        """
        chain_type = predict_object.script_arguments.chain_type
        args = predict_object.script_arguments

        self.log(f"Loading Data Config")
        data_config_library = DataConfigLibrary(custom_heavy_data_config=args.heavy_data_config,
                                                custom_kappa_data_config=args.kappa_data_config,
                                                custom_lambda_data_config=args.lambda_data_config)
        data_config_library.mount_type(chain_type)
        predict_object.data_config_library = data_config_library
        predict_object.mount_genotype_list()
        self.log("Data Config loaded successfully, Type Mounted : {}".format(chain_type))

        return predict_object

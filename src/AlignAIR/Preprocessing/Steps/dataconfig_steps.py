from AlignAIR.Step.Step import Step
from GenAIRR.data import builtin_kappa_chain_data_config, builtin_lambda_chain_data_config, \
    builtin_heavy_chain_data_config
import pickle
class ConfigLoadStep(Step):
    def __init__(self, name, logger=None):
        super().__init__(name, logger)

    def process(self, predict_object):
        """
        Loads configuration based on the chain type from provided paths.

        Args:
            predict_object (PredictObject): The object that holds the chain_type and config_paths.

        Returns:
            PredictObject: Updated with loaded configuration.
        """
        chain_type = predict_object.script_arguments.chain_type
        args = predict_object.script_arguments
        config_paths ={'heavy': args.heavy_data_config,
                    'kappa': args.kappa_data_config,
                    'lambda': args.lambda_data_config
                    }

        self.log(f"Loading Data Config for {chain_type}")

        if chain_type == 'heavy':
            if config_paths['heavy'] == 'D':
                config = {'heavy': builtin_heavy_chain_data_config()}
            else:
                with open(config_paths['heavy'], 'rb') as h:
                    config = {'heavy': pickle.load(h)}

        elif chain_type == 'light':
            config = {}
            if config_paths['kappa'] == 'D':
                config['kappa'] = builtin_kappa_chain_data_config()
            else:
                with open(config_paths['kappa'], 'rb') as h:
                    config['kappa'] = pickle.load(h)

            if config_paths['lambda'] == 'D':
                config['lambda'] = builtin_lambda_chain_data_config()
            else:
                with open(config_paths['lambda'], 'rb') as h:
                    config['lambda'] = pickle.load(h)
        else:
            raise ValueError(f'Unknown Chain Type: {chain_type}')

        self.log("Data Config loaded successfully")
        predict_object.data_config = config
        return predict_object

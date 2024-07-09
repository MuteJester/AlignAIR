import multiprocessing
import pickle
from multiprocessing import Process

from AlignAIR.Data import HeavyChainDataset, LightChainDataset
from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
from AlignAIR.Models.LightChain import LightChainAlignAIRR
from AlignAIR.PretrainedComponents import builtin_orientation_classifier
from AlignAIR.Step.Step import Step
from AlignAIR.Trainers import Trainer
from AlignAIR.Utilities.consumer_producer import READER_WORKER_TYPES




class ModelLoadingStep(Step):
    def __init__(self, name, logger=None):
        super().__init__(name, logger)

    def load_model(self,sequences, chain_type, model_checkpoint, max_sequence_size, config=None):
        if chain_type == 'heavy':
            dataset = HeavyChainDataset(data_path=sequences,
                                        dataconfig=config['heavy'], batch_read_file=True,
                                        max_sequence_length=max_sequence_size)
        elif chain_type == 'light':
            dataset = LightChainDataset(data_path=sequences,
                                        lambda_dataconfig=config['lambda'],
                                        kappa_dataconfig=config['kappa'],
                                        batch_read_file=True, max_sequence_length=max_sequence_size)
        else:
            raise ValueError(f'Unknown Chain Type: {chain_type}')

        trainer = Trainer(
            model=LightChainAlignAIRR if chain_type == 'light' else HeavyChainAlignAIRR,
            dataset=dataset,
            epochs=1,
            steps_per_epoch=1,
            verbose=1,
        )

        trainer.model.build({'tokenized_sequence': (max_sequence_size, 1)})

        MODEL_CHECKPOINT = model_checkpoint

        trainer.model.load_weights(MODEL_CHECKPOINT)

        self.log(f"Loading: {MODEL_CHECKPOINT.split('/')[-1]}")

        self.log(f"Model Loaded Successfully")

        return trainer.model


    def process(self, predict_object):
        return predict_object

    def execute(self, predict_object):
        self.log("Loading main model...")
        predict_object.model = self.load_model(
            predict_object.script_arguments.sequences,
            predict_object.script_arguments.chain_type,
            predict_object.script_arguments.model_checkpoint,
            predict_object.script_arguments.max_input_size,
            predict_object.data_config
        )
        self.log("Main Model Loaded...")

        # Load DNA orientation model if required
        if predict_object.script_arguments.fix_orientation:
            self.log("Loading Orientation Pipeline...")
            path = predict_object.script_arguments.custom_orientation_pipeline_path
            if path:
                with open(path, 'rb') as h:
                    predict_object.orientation_pipeline = pickle.load(h)
            else:
                predict_object.orientation_pipeline = builtin_orientation_classifier()
            self.log('Orientation Pipeline Loaded Successfully')

        return predict_object

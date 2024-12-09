import multiprocessing
import pickle
from multiprocessing import Process

import torch
from torch.utils.data import DataLoader

from AlignAIR.Data import HeavyChainDataset, LightChainDataset
from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
from AlignAIR.Models.LightChain import LightChainAlignAIRR
#from AlignAIR.Data import HeavyChainDataset, LightChainDataset
#from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
#from AlignAIR.Models.LightChain import LightChainAlignAIRR
from AlignAIR.Preprocessing.LongSequence.FastKmerDensityExtractor import FastKmerDensityExtractor
from AlignAIR.PretrainedComponents import builtin_orientation_classifier
from AlignAIR.Pytorch.Dataset import CSVReaderDataset
from AlignAIR.Pytorch.HeavyChainAlignAIR import HeavyChainAlignAIR
from AlignAIR.Pytorch.InputPreProcessors import HeavyChainInputPreProcessor
from AlignAIR.Pytorch.Loss import AlignAIRHeavyChainLoss
from AlignAIR.Pytorch.Trainer import AlignAIRTrainer
from AlignAIR.Step.Step import Step
from AlignAIR.Trainers import Trainer
# from AlignAIR.Trainers import Trainer
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

        model_params = dataset.generate_model_params()

        model = LightChainAlignAIRR if chain_type == 'light' else HeavyChainAlignAIRR
        model = model(**model_params)
        model.build({'tokenized_sequence': (max_sequence_size, 1)})
        MODEL_CHECKPOINT = model_checkpoint
        model.load_weights(MODEL_CHECKPOINT)
        self.log(f"Loading: {MODEL_CHECKPOINT.split('/')[-1]}")
        self.log(f"Model Loaded Successfully")

        return model


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

        self.log("Loading Fitting Kmer Density Model...")

        ref_alleles = None
        if predict_object.chain_type == 'heavy':
            data_config = predict_object.data_config[predict_object.chain_type]
            hc_alleles = [i.ungapped_seq.upper() for j in data_config.v_alleles for i in data_config.v_alleles[j]]
            hc_alleles += [i.ungapped_seq.upper() for j in data_config.j_alleles for i in data_config.j_alleles[j]]
            hc_alleles += [i.ungapped_seq.upper() for j in data_config.d_alleles for i in data_config.d_alleles[j]]
            ref_alleles = hc_alleles
        elif predict_object.chain_type == 'light':
            data_config_lambda = predict_object.data_config['lambda']
            data_config_kappa = predict_object.data_config['kappa']

            lc_alleles = [i.ungapped_seq.upper() for j in data_config_lambda.v_alleles for i in
                          data_config_lambda.v_alleles[j]] + \
                         [i.ungapped_seq.upper() for j in data_config_lambda.v_alleles for i in
                          data_config_lambda.v_alleles[j]]
            lc_alleles += [i.ungapped_seq.upper() for j in data_config_kappa.j_alleles for i in
                           data_config_kappa.j_alleles[j]] + \
                          [i.ungapped_seq.upper() for j in data_config_kappa.j_alleles for i in
                           data_config_kappa.j_alleles[j]]
            ref_alleles = lc_alleles

        predict_object.candidate_sequence_extractor = FastKmerDensityExtractor(11, max_length=576, allowed_mismatches=0)
        predict_object.candidate_sequence_extractor.fit(ref_alleles)
        self.log("Kmer Density Model Fitted Successfully...")
        return predict_object


class PytorchModelLoadingStep(Step):
    def __init__(self, name, logger=None):
        super().__init__(name, logger)

    def load_model(self,sequences, chain_type, model_checkpoint, max_sequence_size, config=None):
        if chain_type == 'heavy':


            pre_processor = HeavyChainInputPreProcessor(heavy_chain_dataconfig=config['heavy'])
            dataset = CSVReaderDataset(
                csv_file=sequences,
                preprocessor=pre_processor,  # Custom preprocessing logic
                batch_size=64,
                separator=','
            )

        elif chain_type == 'light':
            # TO DO
            pass
        else:
            raise ValueError(f'Unknown Chain Type: {chain_type}')

        v_allele_count = pre_processor.v_allele_count
        d_allele_count = pre_processor.d_allele_count
        j_allele_count = pre_processor.j_allele_count

        # 3. Initialize Model
        model = HeavyChainAlignAIR(
            max_seq_length=576,
            v_allele_count=v_allele_count,
            d_allele_count=d_allele_count,
            j_allele_count=j_allele_count
        )

        # 4. Define Loss Function
        loss_function = AlignAIRHeavyChainLoss

        # 5. Instantiate Trainer
        trainer = AlignAIRTrainer(
            model=model,
            dataset=dataset,
            loss_function=loss_function,
            optimizer=torch.optim.Adam,
            optimizer_params={"lr": 1e-3},
            epochs=10,
            batch_size=64,
            log_to_file=True,
            log_file_path="C:/Users/tomas/Downloads/",
            verbose=2,
            batches_per_epoch=64
        )


        MODEL_CHECKPOINT = model_checkpoint

        trainer.load_model(MODEL_CHECKPOINT)

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

        self.log("Loading Fitting Kmer Density Model...")

        ref_alleles = None
        if predict_object.chain_type == 'heavy':
            data_config = predict_object.data_config[predict_object.chain_type]
            hc_alleles = [i.ungapped_seq.upper() for j in data_config.v_alleles for i in data_config.v_alleles[j]]
            hc_alleles += [i.ungapped_seq.upper() for j in data_config.j_alleles for i in data_config.j_alleles[j]]
            hc_alleles += [i.ungapped_seq.upper() for j in data_config.d_alleles for i in data_config.d_alleles[j]]
            ref_alleles = hc_alleles
        elif predict_object.chain_type == 'light':
            data_config_lambda = predict_object.data_config['lambda']
            data_config_kappa = predict_object.data_config['kappa']

            lc_alleles = [i.ungapped_seq.upper() for j in data_config_lambda.v_alleles for i in
                          data_config_lambda.v_alleles[j]] + \
                         [i.ungapped_seq.upper() for j in data_config_lambda.v_alleles for i in
                          data_config_lambda.v_alleles[j]]
            lc_alleles += [i.ungapped_seq.upper() for j in data_config_kappa.j_alleles for i in
                           data_config_kappa.j_alleles[j]] + \
                          [i.ungapped_seq.upper() for j in data_config_kappa.j_alleles for i in
                           data_config_kappa.j_alleles[j]]
            ref_alleles = lc_alleles

        predict_object.candidate_sequence_extractor = FastKmerDensityExtractor(11, max_length=576, allowed_mismatches=0)
        predict_object.candidate_sequence_extractor.fit(ref_alleles)
        self.log("Kmer Density Model Fitted Successfully...")
        return predict_object

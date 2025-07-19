import pickle
from GenAIRR.dataconfig import DataConfig

from AlignAIR.Data import SingleChainDataset, MultiChainDataset, MultiDataConfigContainer
from AlignAIR.Models.SingleChainAlignAIR.SingleChainAlignAIR import SingleChainAlignAIR
from AlignAIR.Models.MultiChainAlignAIR.MultiChainAlignAIR import MultiChainAlignAIR
from AlignAIR.Preprocessing.LongSequence.FastKmerDensityExtractor import FastKmerDensityExtractor
from AlignAIR.PretrainedComponents import builtin_orientation_classifier
from AlignAIR.Step.Step import Step
from AlignAIR.Utilities.CustomModelParamsYaml import CustomModelParamsYaml
from AlignAIR.Utilities.step_utilities import FileInfo, MultiFileInfoContainer

BACKEND_ENGINE = 'tensorflow'

class ModelLoadingStep(Step):
    def __init__(self, name):
        super().__init__(name)


    def load_model(self, file_info,
                   dataconfig: MultiDataConfigContainer,
                   model_checkpoint: str,
                   max_sequence_size: int,
                   custom_model_parameters: CustomModelParamsYaml = None):

        # Determine if we're in training mode (multiple files) or prediction mode (single file)
        is_training_mode = hasattr(file_info, 'file_infos') and len(file_info) > 1
        is_multi_chain = len(dataconfig) > 1

        if is_multi_chain:
            if is_training_mode:
                # Training mode: Multiple files, each for different chain types
                data_paths = [fi.path for fi in file_info]
                self.log(f"Multi-chain training mode: {len(data_paths)} files for {len(dataconfig)} configs")
            else:
                # Prediction mode: Single file with mixed sequences to classify
                if hasattr(file_info, 'file_infos'):
                    # MultiFileInfoContainer with single file
                    data_paths = [file_info[0].path]
                else:
                    # Single FileInfo
                    data_paths = [file_info.path]
                self.log(f"Multi-chain prediction mode: 1 file for {len(dataconfig)} configs")
                
            dataset = MultiChainDataset(
                data_paths=data_paths,
                dataconfigs=dataconfig,  # Pass MultiDataConfigContainer directly
                max_sequence_length=max_sequence_size,
            )
        else:
            # Single chain mode - use existing logic
            if hasattr(file_info, 'file_infos'):
                # MultiFileInfoContainer with single file
                data_path = file_info[0].path
            else:
                # Single FileInfo
                data_path = file_info.path
                
            dataset = SingleChainDataset(
                data_path=data_path,
                dataconfig=dataconfig,  # MultiDataConfigContainer acts as proxy for single config
                max_sequence_length=max_sequence_size,
            )

        model_params = dataset.generate_model_params()

        if custom_model_parameters:
            for key in custom_model_parameters.accepted_keys:
                if hasattr(custom_model_parameters, key):
                    model_params[key] = getattr(custom_model_parameters, key)

            self.log(f"Custom Model Parameters: {model_params} loaded successfully...")

        # Choose appropriate model class based on whether it's multi-chain or single-chain
        if is_multi_chain:
            # Use MultiChainAlignAIR for multi-chain scenarios
            model_params['dataconfigs'] = dataconfig  # Pass the MultiDataConfigContainer
            model = MultiChainAlignAIR(**model_params)
            self.log("Using MultiChainAlignAIR model")
        else:
            # Use SingleChainAlignAIR for single-chain scenarios
            model = SingleChainAlignAIR(**model_params)
            self.log("Using SingleChainAlignAIR model")

        model.build({'tokenized_sequence': (max_sequence_size, 1)})
        model.load_weights(model_checkpoint)
        self.log(f"Loading: {model_checkpoint.split('/')[-1]}")
        self.log(f"Model Loaded Successfully")

        return model

    def process(self, predict_object):
        return predict_object

    def execute(self, predict_object):
        self.log("Loading main model...")

        if predict_object.script_arguments.finetuned_model_params_yaml:
            self.log("Custom model parameters detected... Assuming custom finetuned model is being loaded...")
            custom_model_parameters = CustomModelParamsYaml(predict_object.script_arguments.finetuned_model_params_yaml)
            self.log("Custom model parameters loaded successfully...")
        else:
            custom_model_parameters = None


        predict_object.model = self.load_model(
            predict_object.file_info,
            predict_object.dataconfig,
            predict_object.script_arguments.model_checkpoint,
            predict_object.script_arguments.max_input_size,
            custom_model_parameters
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
                # For multi-chain, use the first chain type for orientation
                chain_type = predict_object.dataconfig.chain_types()[0] if len(predict_object.dataconfig) > 1 else predict_object.dataconfig.metadata.chain_type
                predict_object.orientation_pipeline = builtin_orientation_classifier(chain_type)
            self.log('Orientation Pipeline Loaded Successfully')

        self.log("Loading Fitting Kmer Density Model...")

        ref_alleles = []
        
        # Handle both single and multi-chain dataconfigs
        if len(predict_object.dataconfig) == 1:
            # Single chain
            ref_alleles = (
                list(map(lambda x: x.ungapped_seq.upper(), predict_object.dataconfig.allele_list('v'))) +
                list(map(lambda x: x.ungapped_seq.upper(), predict_object.dataconfig.allele_list('j')))
            )
            if predict_object.dataconfig.metadata.has_d:
                ref_alleles += list(map(lambda x: x.ungapped_seq.upper(), predict_object.dataconfig.allele_list('d')))
        else:
            # Multi-chain: aggregate alleles from all configs
            for dc in predict_object.dataconfig:
                ref_alleles.extend(list(map(lambda x: x.ungapped_seq.upper(), dc.allele_list('v'))))
                ref_alleles.extend(list(map(lambda x: x.ungapped_seq.upper(), dc.allele_list('j'))))
                if dc.metadata.has_d:
                    ref_alleles.extend(list(map(lambda x: x.ungapped_seq.upper(), dc.allele_list('d'))))
            
            # Remove duplicates while preserving order
            ref_alleles = list(dict.fromkeys(ref_alleles))


        predict_object.candidate_sequence_extractor = FastKmerDensityExtractor(11, max_length=576, allowed_mismatches=0)
        predict_object.candidate_sequence_extractor.fit(ref_alleles)
        self.log("Kmer Density Model Fitted Successfully...")
        return predict_object


if BACKEND_ENGINE == 'torch':

    import torch
    from AlignAIR.Pytorch.Dataset import CSVReaderDataset
    from AlignAIR.Pytorch.HeavyChainAlignAIR import HeavyChainAlignAIR
    from AlignAIR.Pytorch.InputPreProcessors import HeavyChainInputPreProcessor
    from AlignAIR.Pytorch.Loss import AlignAIRHeavyChainLoss
    from AlignAIR.Pytorch.Trainer import AlignAIRTrainer

    class PytorchModelLoadingStep(Step):
        def __init__(self, name, logger=None):
            super().__init__(name, logger)

        def load_model(self, sequences, chain_type, model_checkpoint, max_sequence_size, config=None):
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

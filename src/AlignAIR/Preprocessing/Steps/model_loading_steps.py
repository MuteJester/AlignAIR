import pickle
from AlignAIR.Data import HeavyChainDataset, LightChainDataset
from AlignAIR.Preprocessing.LongSequence.FastKmerDensityExtractor import FastKmerDensityExtractor
from AlignAIR.PretrainedComponents import builtin_orientation_classifier
from AlignAIR.Step.Step import Step
from AlignAIR.Utilities.CustomModelParamsYaml import CustomModelParamsYaml
from AlignAIR.Utilities.step_utilities import DataConfigLibrary, FileInfo

BACKEND_ENGINE = 'tensorflow'

class ModelLoadingStep(Step):
    def __init__(self, name):
        super().__init__(name)

    @staticmethod
    def get_dataset_object(file_info: FileInfo, data_config_library: DataConfigLibrary, max_sequence_size):
        dataset_object = data_config_library.matching_dataset_object
        if data_config_library.mounted in ['heavy']:
            dataset = dataset_object(data_path=file_info.path,
                                        dataconfig=data_config_library.config(), batch_read_file=True,
                                        max_sequence_length=max_sequence_size)
        elif data_config_library.mounted in ['tcrb']:
            dataset = dataset_object(data_path=file_info.path,
                                     dataconfig=data_config_library.config('tcrb'), batch_read_file=True,
                                     max_sequence_length=max_sequence_size)

        elif data_config_library.mounted == 'light':
            dataset = dataset_object(data_path=file_info.path,
                                        lambda_dataconfig=data_config_library.config('lambda'),
                                        kappa_dataconfig=data_config_library.config('kappa'),
                                        batch_read_file=True, max_sequence_length=max_sequence_size)
        else:
            raise ValueError(f'Unknown Chain Type: {data_config_library.mounted}')
        return dataset

    def load_model(self, file_info: FileInfo,
                   data_config_library: DataConfigLibrary,
                   model_checkpoint: str,
                   max_sequence_size: int,
                   custom_model_parameters: CustomModelParamsYaml = None):

        dataset = self.get_dataset_object(file_info, data_config_library, max_sequence_size)

        model_params = dataset.generate_model_params()

        if custom_model_parameters:
            for key in custom_model_parameters.accepted_keys:
                if hasattr(custom_model_parameters, key):
                    model_params[key] = getattr(custom_model_parameters, key)


            self.log(f"Custom Model Parameters: {model_params} loaded successfully...")


        model = data_config_library.matching_alignair_model(**model_params)

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
            predict_object.data_config_library,
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
                predict_object.orientation_pipeline = builtin_orientation_classifier(predict_object.data_config_library.mounted)
            self.log('Orientation Pipeline Loaded Successfully')

        self.log("Loading Fitting Kmer Density Model...")

        ref_alleles = None
        ref_alleles = (
            predict_object.data_config_library.reference_allele_sequences('v')+
            predict_object.data_config_library.reference_allele_sequences('d')+
            predict_object.data_config_library.reference_allele_sequences('j')
        )


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

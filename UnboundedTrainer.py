import pickle
from uuid import uuid4
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from SequenceCorruptor import global_genotype
from VDeepJDataset import VDeepJDataset
from VDeepJModelExperimental import VDeepJAllignExperimentalV5
from VDeepJUnbondedDataset import VDeepJUnbondedDataset, VDeepJUnbondedDatasetSingleBeam


class UnboundedTrainer:

    def __init__(
            self,
            model,
            epochs,
            batch_size,
            steps_per_epoch,
            input_size=512,
            N_proportion=0.02,
            randomize_rate=False,
            log_to_file=False,
            log_file_name=None,
            log_file_path=None,
            corrupt_beginning=False,
            classification_head_metric='categorical_accuracy',
            interval_head_metric='mae',
            corrupt_proba=1,
            verbose=0,
            nucleotide_add_coef=35,
            nucleotide_remove_coef=50,
            random_sequence_add_proba=1,
            single_base_stream_proba=0,
            duplicate_leading_proba=0,
            random_allele_proba=0,
            num_parallel_calls=1,
            use_gene_masking=False,
            batch_file_reader=False,
            pretrained=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=tf.keras.optimizers.Adam,
            optimizers_params=None
    ):

        self.pretrained = pretrained
        self.model = model
        self.model_type = model
        self.epochs = epochs
        self.input_size = input_size
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.N_proportion = N_proportion
        self.corrupt_beginning = corrupt_beginning
        self.randomize_rate = randomize_rate
        self.classification_head_metric = classification_head_metric
        self.interval_head_metric = interval_head_metric
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.optimizers = optimizers
        self.optimizers_params = optimizers_params
        self.history = None
        self.num_parallel_calls = num_parallel_calls
        self.verbose = verbose
        self.log_file_name = log_file_name
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path
        self.use_gene_masking = use_gene_masking
        self.batch_file_reader = batch_file_reader,
        self.corrupt_proba = corrupt_proba
        self.nucleotide_add_coef = nucleotide_add_coef
        self.nucleotide_remove_coef = nucleotide_remove_coef

        self.train_dataset = VDeepJUnbondedDataset(
            max_sequence_length=self.input_size,
            corrupt_beginning=self.corrupt_beginning,
            corrupt_proba=self.corrupt_proba,
            nucleotide_add_coef=self.nucleotide_add_coef,
            nucleotide_remove_coef=self.nucleotide_remove_coef,
            batch_size=self.batch_size,
            randomize_rate=randomize_rate,
            N_proportion=N_proportion,
            random_sequence_add_proba=random_sequence_add_proba,
            single_base_stream_proba=single_base_stream_proba,
            duplicate_leading_proba=duplicate_leading_proba,
            random_allele_proba=random_allele_proba
        )

        # Set Up Trainer Instance
        self._load_pretrained_model()  # only if pretrained is not None
        self._compile_model()

    def _load_pretrained_model(self):
        model_params = self.train_dataset.generate_model_params()
        model_params['use_gene_masking'] = self.use_gene_masking
        self.model = self.model_type(**model_params)
        if self.pretrained is not None:
            self.model.load_weights(self.pretrained)

    def _compile_model(self):
        metrics = {key: self.interval_head_metric for key in
                   ['v_start', 'v_end', 'd_start', 'd_end', 'j_start', 'j_end']}
        for key in ['v_family',
                    'v_gene',
                    'v_allele',
                    'd_family',
                    'd_gene',
                    'd_allele',
                    'j_gene',
                    'j_allele']:
            metrics[key] = self.classification_head_metric

        if self.optimizers_params is not None:
            self.model.compile(optimizer=self.optimizers(**self.optimizers_params),
                               loss=None,
                               metrics=metrics)
        else:
            self.model.compile(optimizer=self.optimizers(),
                               loss=None,
                               metrics=metrics)

    def train(self):

        _callbacks = [] if self.callbacks is None else self.callbacks
        if self.log_to_file:
            if self.log_file_path is None:
                raise ValueError('No log_file_path was given to Trainer')

            file_name = str(uuid4()) + '.csv' if self.log_to_file is None else self.log_file_name
            csv_logger = CSVLogger(self.log_file_path + file_name + '.csv', append=True, separator=',')
            _callbacks.append(csv_logger)

        train_dataset = self.train_dataset.get_train_dataset()
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        def preprocess_data(*args):
            return args

        train_dataset = train_dataset.map(
            preprocess_data,
            num_parallel_calls=self.num_parallel_calls
        )

        self.history = self.model.fit(
            train_dataset,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch // self.batch_size,
            verbose=self.verbose,
            callbacks=_callbacks

        )

    def save_model(self, path):
        postfix = str(uuid4())
        self.model.save_weights(path + f'VDeepJModel_{postfix}_weights')
        print(f'Model Saved!\n Location: {path + f"VDeepJModel_{postfix}_weights"}')
        return path + f'VDeepJModel_{postfix}_weights'

    def save_dataset_object(self, path):
        with open(path, 'wb') as h:
            pickle.dump(self.train_dataset, h)
        print(f'Dataset Object Saved at {path}')

    def load_model(self, weights_path):
        self.model.load_weights(weights_path)

    def load_dataset_object(self, path):
        with open(path, 'rb') as h:
            self.train_dataset = pickle.load(h)

    def load_dataset(self, dataset_path):
        self.train_dataset = VDeepJDataset(data_path=dataset_path,
                                           max_sequence_length=self.input_size,
                                           corrupt_beginning=self.corrupt_beginning,
                                           corrupt_proba=self.corrupt_proba,
                                           nucleotide_add_coef=self.nucleotide_add_coef,
                                           nucleotide_remove_coef=self.nucleotide_remove_coef,
                                           batch_size=self.batch_size)
        # Set Up Trainer Instance
        self._load_pretrained_model()  # only if pretrained is not None
        self._compile_model()

    def rebuild_model(self):
        self.model = self.model_type(**self.train_dataset.generate_model_params())

    def set_custom_dataset_object(self, dataset_instance):
        self.train_dataset = dataset_instance

    def plot_model(self):
        self.model.plot_model((self.input_size, 1))

    def model_summary(self):
        self.model.model_summary((self.input_size, 1))

    def plot_history(self, write_path=None):
        num_plots = len(self.history.history)
        num_cols = 3
        num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows based on num_plots

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 5 * num_rows))

        for i, column in enumerate(list(self.history.history)):
            row = i // num_cols  # Calculate the row index for the current subplot
            col = i % num_cols  # Calculate the column index for the current subplot

            ax = axes[row, col] if num_rows > 1 else axes[col]  # Handle single row case

            ax.plot(self.history.epoch, self.history.history[column])
            ax.set_xlabel('Epoch')
            # ax.set_ylabel(column)
            ax.set_title(column)
            ax.grid(lw=2, ls=':')

        # Remove any empty subplots
        if num_plots < num_rows * num_cols:
            for i in range(num_plots, num_rows * num_cols):
                row = i // num_cols
                col = i % num_cols
                fig.delaxes(axes[row, col])

        plt.tight_layout()
        if write_path is None:
            plt.show()
        else:
            plt.savefig(write_path, dpi=300, facecolor='white')


class SingleBeamUnboundedTrainer:

    def __init__(
            self,
            model,
            epochs,
            batch_size,
            steps_per_epoch,
            input_size=512,
            N_proportion=0.02,
            randomize_rate=False,
            log_to_file=False,
            log_file_name=None,
            log_file_path=None,
            corrupt_beginning=False,
            classification_head_metric='categorical_accuracy',
            interval_head_metric='mae',
            corrupt_proba=1,
            verbose=0,
            airrship_mutation_rate=0.2,
            nucleotide_add_coef=35,
            nucleotide_remove_coef=50,
            random_sequence_add_proba=1,
            single_base_stream_proba=0,
            duplicate_leading_proba=0,
            random_allele_proba=0,
            num_parallel_calls=1,
            batch_file_reader=False,
            pretrained=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=tf.keras.optimizers.Adam,
            optimizers_params=None
    ):

        self.pretrained = pretrained
        self.model = model
        self.model_type = model
        self.epochs = epochs
        self.input_size = input_size
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.N_proportion = N_proportion
        self.corrupt_beginning = corrupt_beginning
        self.randomize_rate = randomize_rate
        self.classification_head_metric = classification_head_metric
        self.interval_head_metric = interval_head_metric
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.optimizers = optimizers
        self.optimizers_params = optimizers_params
        self.history = None
        self.num_parallel_calls = num_parallel_calls
        self.verbose = verbose
        self.log_file_name = log_file_name
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path
        self.batch_file_reader = batch_file_reader,
        self.corrupt_proba = corrupt_proba
        self.nucleotide_add_coef = nucleotide_add_coef
        self.nucleotide_remove_coef = nucleotide_remove_coef

        self.train_dataset = VDeepJUnbondedDatasetSingleBeam(
            max_sequence_length=self.input_size,
            corrupt_beginning=self.corrupt_beginning,
            corrupt_proba=self.corrupt_proba,
            nucleotide_add_coef=self.nucleotide_add_coef,
            nucleotide_remove_coef=self.nucleotide_remove_coef,
            batch_size=self.batch_size,
            randomize_rate=randomize_rate,
            mutation_rate=airrship_mutation_rate,
            N_proportion=N_proportion,
            random_sequence_add_proba=random_sequence_add_proba,
            single_base_stream_proba=single_base_stream_proba,
            duplicate_leading_proba=duplicate_leading_proba,
            random_allele_proba=random_allele_proba
        )

        # Set Up Trainer Instance
        self._load_pretrained_model()  # only if pretrained is not None
        self._compile_model()

    def _load_pretrained_model(self):
        model_params = self.train_dataset.generate_model_params()
        if self.model_type == VDeepJAllignExperimentalV5:
            Vs = global_genotype()[0]
            self.V_REFERENCE = [i.ungapped_seq.upper() for i in Vs['V']]
            self.V_REFERENCE = self.train_dataset.tokenize_sequences(self.V_REFERENCE)
            model_params['V_REF'] = self.V_REFERENCE
        self.model = self.model_type(**model_params)
        if self.pretrained is not None:
            self.model.load_weights(self.pretrained)

    def _compile_model(self):
        metrics = {key: self.interval_head_metric for key in
                   ['v_start', 'v_end', 'd_start', 'd_end', 'j_start', 'j_end']}

        if type(self.classification_head_metric) == list:
            for key, m in zip(['v_allele',
                               'd_allele',
                               'j_allele'], self.classification_head_metric):
                metrics[key] = m
        else:

            for key in ['v_allele',
                        'd_allele',
                        'j_allele']:
                metrics[key] = self.classification_head_metric

        if self.optimizers_params is not None:
            self.model.compile(optimizer=self.optimizers(**self.optimizers_params),
                               loss=None,
                               metrics=metrics)
        else:
            self.model.compile(optimizer=self.optimizers(),
                               loss=None,
                               metrics=metrics)

    def train(self):

        _callbacks = [] if self.callbacks is None else self.callbacks
        if self.log_to_file:
            if self.log_file_path is None:
                raise ValueError('No log_file_path was given to Trainer')

            file_name = str(uuid4()) + '.csv' if self.log_to_file is None else self.log_file_name
            csv_logger = CSVLogger(self.log_file_path + file_name + '.csv', append=True, separator=',')
            _callbacks.append(csv_logger)

        train_dataset = self.train_dataset.get_train_dataset()
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        def preprocess_data(*args):
            return args

        train_dataset = train_dataset.map(
            preprocess_data,
            num_parallel_calls=self.num_parallel_calls
        )

        self.history = self.model.fit(
            train_dataset,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch // self.batch_size,
            verbose=self.verbose,
            callbacks=_callbacks

        )

    def save_model(self, path):
        postfix = str(uuid4())
        self.model.save_weights(path + f'VDeepJModel_{postfix}_weights')
        print(f'Model Saved!\n Location: {path + f"VDeepJModel_{postfix}_weights"}')
        return path + f'VDeepJModel_{postfix}_weights'

    def save_dataset_object(self, path):
        with open(path, 'wb') as h:
            pickle.dump(self.train_dataset, h)
        print(f'Dataset Object Saved at {path}')

    def load_model(self, weights_path):
        self.model.load_weights(weights_path)

    def load_dataset_object(self, path):
        with open(path, 'rb') as h:
            self.train_dataset = pickle.load(h)

    def load_dataset(self, dataset_path):
        self.train_dataset = VDeepJDataset(data_path=dataset_path,
                                           max_sequence_length=self.input_size,
                                           corrupt_beginning=self.corrupt_beginning,
                                           corrupt_proba=self.corrupt_proba,
                                           nucleotide_add_coef=self.nucleotide_add_coef,
                                           nucleotide_remove_coef=self.nucleotide_remove_coef,
                                           batch_size=self.batch_size)
        # Set Up Trainer Instance
        self._load_pretrained_model()  # only if pretrained is not None
        self._compile_model()

    def rebuild_model(self):
        self.model = self.model_type(**self.train_dataset.generate_model_params())

    def set_custom_dataset_object(self, dataset_instance):
        self.train_dataset = dataset_instance

    def plot_model(self):
        self.model.plot_model((self.input_size, 1))

    def model_summary(self):
        self.model.model_summary((self.input_size, 1))

    def plot_history(self, write_path=None):
        num_plots = len(self.history.history)
        num_cols = 3
        num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows based on num_plots

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 5 * num_rows))

        for i, column in enumerate(list(self.history.history)):
            row = i // num_cols  # Calculate the row index for the current subplot
            col = i % num_cols  # Calculate the column index for the current subplot

            ax = axes[row, col] if num_rows > 1 else axes[col]  # Handle single row case

            ax.plot(self.history.epoch, self.history.history[column])
            ax.set_xlabel('Epoch')
            # ax.set_ylabel(column)
            ax.set_title(column)
            ax.grid(lw=2, ls=':')

        # Remove any empty subplots
        if num_plots < num_rows * num_cols:
            for i in range(num_plots, num_rows * num_cols):
                row = i // num_cols
                col = i % num_cols
                fig.delaxes(axes[row, col])

        plt.tight_layout()
        if write_path is None:
            plt.show()
        else:
            plt.savefig(write_path, dpi=300, facecolor='white')


class SingleBeamUnboundedTrainerMP:

    def __init__(
            self,
            model,
            epochs,
            batch_size,
            steps_per_epoch,
            data_gen,
            input_size=512,
            N_proportion=0.02,
            randomize_rate=False,
            log_to_file=False,
            log_file_name=None,
            log_file_path=None,
            corrupt_beginning=False,
            classification_head_metric='categorical_accuracy',
            interval_head_metric='mae',
            corrupt_proba=1,
            verbose=0,
            airrship_mutation_rate=0.2,
            nucleotide_add_coef=35,
            nucleotide_remove_coef=50,
            random_sequence_add_proba=1,
            single_base_stream_proba=0,
            duplicate_leading_proba=0,
            random_allele_proba=0,
            num_parallel_calls=1,
            batch_file_reader=False,
            pretrained=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=tf.keras.optimizers.Adam,
            optimizers_params=None
    ):

        self.pretrained = pretrained
        self.model = model
        self.model_type = model
        self.epochs = epochs
        self.input_size = input_size
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.N_proportion = N_proportion
        self.corrupt_beginning = corrupt_beginning
        self.randomize_rate = randomize_rate
        self.classification_head_metric = classification_head_metric
        self.interval_head_metric = interval_head_metric
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.optimizers = optimizers
        self.optimizers_params = optimizers_params
        self.history = None
        self.num_parallel_calls = num_parallel_calls
        self.verbose = verbose
        self.log_file_name = log_file_name
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path
        self.batch_file_reader = batch_file_reader,
        self.corrupt_proba = corrupt_proba
        self.nucleotide_add_coef = nucleotide_add_coef
        self.nucleotide_remove_coef = nucleotide_remove_coef

        self.data_generator = data_gen

        self.train_dataset = VDeepJUnbondedDatasetSingleBeam(
            max_sequence_length=self.input_size,
            corrupt_beginning=self.corrupt_beginning,
            corrupt_proba=self.corrupt_proba,
            nucleotide_add_coef=self.nucleotide_add_coef,
            nucleotide_remove_coef=self.nucleotide_remove_coef,
            batch_size=self.batch_size,
            randomize_rate=randomize_rate,
            mutation_rate=airrship_mutation_rate,
            N_proportion=N_proportion,
            random_sequence_add_proba=random_sequence_add_proba,
            single_base_stream_proba=single_base_stream_proba,
            duplicate_leading_proba=duplicate_leading_proba,
            random_allele_proba=random_allele_proba
        )
        # Set Up Trainer Instance
        self._load_pretrained_model()  # only if pretrained is not None
        self._compile_model()

    def _load_pretrained_model(self):
        model_params = self.train_dataset.generate_model_params()
        if self.model_type == VDeepJAllignExperimentalV5:
            Vs = global_genotype()[0]
            self.V_REFERENCE = [i.ungapped_seq.upper() for i in Vs['V']]
            self.V_REFERENCE = self.train_dataset.tokenize_sequences(self.V_REFERENCE)
            model_params['V_REF'] = self.V_REFERENCE
        self.model = self.model_type(**model_params)
        if self.pretrained is not None:
            self.model.load_weights(self.pretrained)

    def _compile_model(self):
        metrics = {key: self.interval_head_metric for key in
                   ['v_start', 'v_end', 'd_start', 'd_end', 'j_start', 'j_end']}

        if type(self.classification_head_metric) == list:
            for key, m in zip(['v_allele',
                               'd_allele',
                               'j_allele'], self.classification_head_metric):
                metrics[key] = m
        else:

            for key in ['v_allele',
                        'd_allele',
                        'j_allele']:
                metrics[key] = self.classification_head_metric

        if self.optimizers_params is not None:
            self.model.compile(optimizer=self.optimizers(**self.optimizers_params),
                               loss=None,
                               metrics=metrics)
        else:
            self.model.compile(optimizer=self.optimizers(),
                               loss=None,
                               metrics=metrics)

    def train(self):

        _callbacks = [] if self.callbacks is None else self.callbacks
        if self.log_to_file:
            if self.log_file_path is None:
                raise ValueError('No log_file_path was given to Trainer')

            file_name = str(uuid4()) + '.csv' if self.log_to_file is None else self.log_file_name
            csv_logger = CSVLogger(self.log_file_path + file_name + '.csv', append=True, separator=',')
            _callbacks.append(csv_logger)

        generator = self.data_generator

        self.history = self.model.fit(
            generator,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch // self.batch_size,
            verbose=self.verbose,
            callbacks=_callbacks

        )

    def save_model(self, path):
        postfix = str(uuid4())
        self.model.save_weights(path + f'VDeepJModel_{postfix}_weights')
        print(f'Model Saved!\n Location: {path + f"VDeepJModel_{postfix}_weights"}')
        return path + f'VDeepJModel_{postfix}_weights'

    def save_dataset_object(self, path):
        with open(path, 'wb') as h:
            pickle.dump(self.train_dataset, h)
        print(f'Dataset Object Saved at {path}')

    def load_model(self, weights_path):
        self.model.load_weights(weights_path)

    def load_dataset_object(self, path):
        with open(path, 'rb') as h:
            self.train_dataset = pickle.load(h)

    def load_dataset(self, dataset_path):
        self.train_dataset = VDeepJDataset(data_path=dataset_path,
                                           max_sequence_length=self.input_size,
                                           corrupt_beginning=self.corrupt_beginning,
                                           corrupt_proba=self.corrupt_proba,
                                           nucleotide_add_coef=self.nucleotide_add_coef,
                                           nucleotide_remove_coef=self.nucleotide_remove_coef,
                                           batch_size=self.batch_size)
        # Set Up Trainer Instance
        self._load_pretrained_model()  # only if pretrained is not None
        self._compile_model()

    def rebuild_model(self):
        self.model = self.model_type(**self.train_dataset.generate_model_params())

    def set_custom_dataset_object(self, dataset_instance):
        self.train_dataset = dataset_instance

    def plot_model(self):
        self.model.plot_model((self.input_size, 1))

    def model_summary(self):
        self.model.model_summary((self.input_size, 1))

    def plot_history(self, write_path=None):
        num_plots = len(self.history.history)
        num_cols = 3
        num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows based on num_plots

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 5 * num_rows))

        for i, column in enumerate(list(self.history.history)):
            row = i // num_cols  # Calculate the row index for the current subplot
            col = i % num_cols  # Calculate the column index for the current subplot

            ax = axes[row, col] if num_rows > 1 else axes[col]  # Handle single row case

            ax.plot(self.history.epoch, self.history.history[column])
            ax.set_xlabel('Epoch')
            # ax.set_ylabel(column)
            ax.set_title(column)
            ax.grid(lw=2, ls=':')

        # Remove any empty subplots
        if num_plots < num_rows * num_cols:
            for i in range(num_plots, num_rows * num_cols):
                row = i // num_cols
                col = i % num_cols
                fig.delaxes(axes[row, col])

        plt.tight_layout()
        if write_path is None:
            plt.show()
        else:
            plt.savefig(write_path, dpi=300, facecolor='white')



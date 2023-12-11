import pickle
from uuid import uuid4
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from VDeepJDataset import VDeepJDatasetRefactored


class HeavyChainTrainer:
    def __init__(
            self,
            data_path,
            batch_read_file,
            model,
            epochs,
            batch_size,
            steps_per_epoch,
            input_size=512,
            num_parallel_calls=8,
            log_to_file=False,
            log_file_name=None,
            log_file_path=None,
            classification_head_metric='categorical_accuracy',
            interval_head_metric='mae',
            verbose=0,
            batch_file_reader=False,
            pretrained=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=tf.keras.optimizers.Adam,
            optimizers_params=None,
            allele_map_path='/home/bcrlab/thomas/AlignAIRR/'

    ):

        self.pretrained = pretrained
        self.model = model
        self.model_type = model
        self.epochs = epochs
        self.input_size = input_size
        self.num_parallel_calls = num_parallel_calls
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.classification_head_metric = classification_head_metric
        self.segmentation_head_metric = interval_head_metric
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.optimizers = optimizers
        self.optimizers_params = optimizers_params
        self.history = None
        self.verbose = verbose
        self.log_file_name = log_file_name
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path
        self.batch_file_reader = batch_file_reader,
        self.allele_map_path = allele_map_path
        self.data_path = data_path
        self.batch_read_file = batch_read_file

        self.train_dataset = VDeepJDatasetRefactored(
            data_path=data_path,
            max_sequence_length=self.input_size,
            batch_size=self.batch_size,
            batch_read_file=self.batch_read_file,
        )

        # Set Up Trainer Instance
        self._load_pretrained_model()  # only if pretrained is not None
        self._compile_model()

    def _load_pretrained_model(self):
        model_params = self.train_dataset.generate_model_params()
        self.model = self.model_type(**model_params)
        if self.pretrained is not None:
            self.model.load_weights(self.pretrained)

    def _compile_model(self):
        metrics = {key: self.segmentation_head_metric for key in ['v_segment', 'd_segment', 'j_segment']}

        if type(self.classification_head_metric) == list:
            for key, m in zip(['v_allele',
                               'd_allele',
                               'j_allele'], self.classification_head_metric):
                if type(m) == list:
                    for met in m:
                        metrics[key] = met
                else:
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
        self.model.save_weights(path + f'AlignAIRR_{postfix}_weights')
        print(f'Model Saved!\n Location: {path + f"VDeepJModel_{postfix}_weights"}')
        return path + f'AlignAIRR_{postfix}_weights'

    def load_model(self, weights_path):
        self.model.load_weights(weights_path)

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

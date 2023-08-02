import copy
import shutil
from dataclasses import asdict, dataclass, field, fields
from typing import Optional
from uuid import uuid4
from VDeepJDataset import VDeepJDataset
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


class Trainer:

    def __init__(
            self,
            model,
            epochs,
            batch_size,
            train_dataset=None,
            input_size=512,
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
            chunked_read=False,
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
        self.chunked_read = chunked_read
        self.corrupt_beginning = corrupt_beginning
        self.classification_head_metric = classification_head_metric
        self.interval_head_metric = interval_head_metric
        self.train_data = train_dataset
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.optimizers = optimizers
        self.optimizers_params = optimizers_params
        self.history = None
        self.verbose = verbose
        self.log_file_name = log_file_name
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path
        self.use_gene_masking = use_gene_masking
        self.batch_file_reader = batch_file_reader,
        self.corrupt_proba = corrupt_proba
        self.nucleotide_add_coef = nucleotide_add_coef
        self.nucleotide_remove_coef = nucleotide_remove_coef

        if train_dataset is not None:
            self.train_dataset = VDeepJDataset(data_path=self.train_data,
                                               max_sequence_length=self.input_size,
                                               corrupt_beginning=self.corrupt_beginning,
                                               corrupt_proba=self.corrupt_proba,
                                               nucleotide_add_coef=self.nucleotide_add_coef,
                                               nucleotide_remove_coef=self.nucleotide_remove_coef,
                                               batch_size=self.batch_size,
                                               batch_read_file=self.batch_file_reader)
        else:
            print('Keep in Mind no Dataset Was Loaded,\n Make Sure to Use "load_dataset" to Add a Train Dataset')

        if train_dataset is not None:
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
        self.history = self.model.fit(
            train_dataset,
            epochs=self.epochs,
            steps_per_epoch=(self.train_dataset.data_length) // self.batch_size,
            verbose=self.verbose,
            callbacks=_callbacks

        )

    def predict(self, eval_dataset_path, raw=True, top_k=1, account_for_padding=True):

        eval_dataset = pd.read_table(eval_dataset_path, usecols=['sequence'])['sequence'].to_list()
        eval_dataset_ = self.train_dataset.tokenize_sequences(eval_dataset)
        predicted = self.model.predict(
            {'tokenized_sequence': eval_dataset_, 'tokenized_sequence_for_masking': eval_dataset_}
            , batch_size=self.batch_size, verbose=self.verbose)

        # Adjust intervals to account for padding
        if account_for_padding:
            padding_sizes = np.array([(self.input_size - len(i))//2 for i in eval_dataset])
            for key in ['v_start', 'v_end', 'd_start', 'd_end', 'j_start', 'j_end']:
                predicted[key] -= padding_sizes

        if raw:
            return predicted
        else:
            pdf = dict()
            for key in self.train_dataset.reverse_ohe_mapping.keys():
                if top_k == 1:
                    pdf[key] = [self.train_dataset.reverse_ohe_mapping[key][np.argmax(i)] for i in predicted[key]]

                elif top_k > 1:
                    K = top_k if top_k < predicted[key].shape[1] else predicted[key].shape[1]
                    top_k_candidates = []
                    scores = []
                    for row in predicted[key]:
                        sorted_ = np.argsort(row)[::-1][:K]
                        candidates = [self.train_dataset.reverse_ohe_mapping[key][c] for c in sorted_]
                        top_k_candidates.append('|'.join(candidates))
                        scores.append(row[sorted_].tolist())

                    pdf[key] = top_k_candidates
                    pdf[key + '_scores'] = scores

            for column in set(predicted.keys()) - set(self.train_dataset.reverse_ohe_mapping.keys()):
                pdf[column] = predicted[column].astype(int).squeeze()

            if top_k == 1:
                pred_df = pd.DataFrame(pdf)
                pred_df['V'] = pred_df['v_family'] + '-' + pred_df['v_gene'] + '*' + pred_df['v_allele']
                pred_df['D'] = pred_df['d_family'] + '-' + pred_df['d_gene'] + '*' + pred_df['d_allele']
                pred_df['J'] = pred_df['j_gene'] + '*' + pred_df['j_allele']
                return pred_df
            else:
                return pd.DataFrame(pdf)

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

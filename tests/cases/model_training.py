import os
import sys
import json
import pickle
import shutil
import subprocess
import unittest
from pathlib import Path
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch
from importlib import resources

import yaml
import pandas as pd
import numpy as np
import tensorflow as tf

# AlignAIR imports
from AlignAIR.Data.PredictionDataset import PredictionDataset
from AlignAIR.Preprocessing.LongSequence.FastKmerDensityExtractor import FastKmerDensityExtractor
from AlignAIR.Utilities.step_utilities import DataConfigLibrary
from AlignAIR.PostProcessing.HeuristicMatching import HeuristicReferenceMatcher
from AlignAIR.PretrainedComponents import builtin_orientation_classifier
from AlignAIR.Utilities.file_processing import FILE_ROW_COUNTERS
from AlignAIR.Utilities.step_utilities import FileInfo

# Model imports
from src.AlignAIR.Data import HeavyChainDataset, LightChainDataset
from src.AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
from src.AlignAIR.Models.LightChain import LightChainAlignAIRR
from src.AlignAIR.Trainers import Trainer

# Data config imports
from GenAIRR.data import (
    builtin_heavy_chain_data_config,
    builtin_kappa_chain_data_config,
    builtin_lambda_chain_data_config,
    builtin_tcrb_data_config
)

# Preprocessing steps
from src.AlignAIR.Preprocessing.Steps.dataconfig_steps import ConfigLoadStep
from src.AlignAIR.Preprocessing.Steps.file_steps import (
    FileNameExtractionStep,
    FileSampleCounterStep
)
from tests.cases.base import BaseTestCase


class ModelTrainingTests(BaseTestCase):
    """Test cases for model training functionality."""

    def test_heavy_chain_model_training(self):
        """Test heavy chain model training pipeline."""
        train_dataset = HeavyChainDataset(
            data_path=str(self.heavy_chain_dataset_path),
            dataconfig=builtin_heavy_chain_data_config(),
            use_streaming=True,
            max_sequence_length=576
        )

        model_params = train_dataset.generate_model_params()
        model = HeavyChainAlignAIRR(**model_params)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(clipnorm=1),
            loss=None,
            metrics={
                'v_start': tf.keras.losses.mse,
                'v_end': tf.keras.losses.mse,
                'd_start': tf.keras.losses.mse,
                'd_end': tf.keras.losses.mse,
                'j_start': tf.keras.losses.mse,
                'j_end': tf.keras.losses.mse,
                'v_allele': tf.keras.losses.binary_crossentropy,
                'd_allele': tf.keras.losses.binary_crossentropy,
                'j_allele': tf.keras.losses.binary_crossentropy,
            }
        )

        trainer = Trainer(
            model=model,
            batch_size=256,
            epochs=1,
            steps_per_epoch=512,
            verbose=1,
            classification_metric=[tf.keras.metrics.AUC()] * 3,
            regression_metric=tf.keras.losses.binary_crossentropy,
        )

        trainer.train(train_dataset)
        self.assertIsNotNone(trainer.history)

    def test_light_chain_model_training(self):
        """Test light chain model training pipeline."""
        train_dataset = LightChainDataset(
            data_path=str(self.light_chain_dataset_path),
            lambda_dataconfig=builtin_lambda_chain_data_config(),
            kappa_dataconfig=builtin_kappa_chain_data_config(),
            use_streaming=True,
            max_sequence_length=576
        )

        model_params = train_dataset.generate_model_params()
        model = LightChainAlignAIRR(**model_params)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(clipnorm=1),
            loss=None,
            metrics={
                'v_start': tf.keras.losses.mse,
                'v_end': tf.keras.losses.mse,
                'j_start': tf.keras.losses.mse,
                'j_end': tf.keras.losses.mse,
                'v_allele': tf.keras.losses.binary_crossentropy,
                'j_allele': tf.keras.losses.binary_crossentropy,
            }
        )

        trainer = Trainer(
            model=model,
            batch_size=256,
            epochs=1,
            steps_per_epoch=512,
            verbose=1,
            classification_metric=[tf.keras.metrics.AUC()] * 3,
            regression_metric=tf.keras.losses.binary_crossentropy,
        )

        trainer.train(train_dataset)
        self.assertIsNotNone(trainer.history)

    def test_tcrb_chain_model_training(self):
        """Test TCR beta chain model training pipeline."""
        train_dataset = HeavyChainDataset(
            data_path=str(self.tcrb_chain_dataset_path),
            dataconfig=builtin_tcrb_data_config(),
            use_streaming=True,
            max_sequence_length=576
        )

        model_params = train_dataset.generate_model_params()
        model = HeavyChainAlignAIRR(**model_params)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(clipnorm=1),
            loss=None,
            metrics={
                'v_start': tf.keras.losses.mse,
                'v_end': tf.keras.losses.mse,
                'd_start': tf.keras.losses.mse,
                'd_end': tf.keras.losses.mse,
                'j_start': tf.keras.losses.mse,
                'j_end': tf.keras.losses.mse,
                'v_allele': tf.keras.losses.binary_crossentropy,
                'd_allele': tf.keras.losses.binary_crossentropy,
                'j_allele': tf.keras.losses.binary_crossentropy,
            }
        )

        trainer = Trainer(
            model=model,
            batch_size=256,
            epochs=1,
            steps_per_epoch=512,
            verbose=1,
            classification_metric=[tf.keras.metrics.AUC()] * 3,
            regression_metric=tf.keras.losses.binary_crossentropy,
        )

        trainer.train(train_dataset)
        self.assertIsNotNone(trainer.history)

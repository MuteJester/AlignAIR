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


class APIScriptTests(BaseTestCase):
    """Integration tests for API scripts."""

    def test_predict_script_heavy_chain(self):
        """Test heavy chain prediction script."""
        args = [
            '--model_checkpoint', str(self.heavy_model_checkpoint),
            '--save_path', f'{self.test_dir}/',
            '--chain_type', 'heavy',
            '--sequences', str(self.heavy_chain_dataset_path),
            '--batch_size', '32',
            '--translate_to_asc'
        ]

        result = self.run_script('AlignAIRRPredict.py', args)
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")

        # Verify output file
        file_name = self.heavy_chain_dataset_path.stem
        output_file = self.test_dir / f'{file_name}_alignairr_results.csv'
        self.assertTrue(output_file.exists(), "Output CSV not created")

        # Validate against expected results
        validation_file = self.test_dir / 'heavychain_predict_validation.csv'
        if validation_file.exists():
            self.assert_csv_equals(output_file, validation_file)

        # Cleanup
        if output_file.exists():
            output_file.unlink()

    def test_predict_script_light_chain(self):
        """Test light chain prediction script."""
        args = [
            '--model_checkpoint', str(self.light_model_checkpoint),
            '--save_path', f'{self.test_dir}/',
            '--chain_type', 'light',
            '--sequences', str(self.light_chain_dataset_path),
            '--batch_size', '32',
            '--translate_to_asc'
        ]

        result = self.run_script('AlignAIRRPredict.py', args)
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")

        # Verify output file
        file_name = self.light_chain_dataset_path.stem
        output_file = self.test_dir / f'{file_name}_alignairr_results.csv'
        self.assertTrue(output_file.exists(), "Output CSV not created")

        # Validate against expected results
        validation_file = self.test_dir / 'lightchain_predict_validation.csv'
        if validation_file.exists():
            self.assert_csv_equals(output_file, validation_file)

        # Cleanup
        if output_file.exists():
            output_file.unlink()

    def test_predict_script_tcrb(self):
        """Test TCR beta chain prediction script."""
        args = [
            '--model_checkpoint', str(self.tcrb_model_checkpoint),
            '--save_path', f'{self.test_dir}/',
            '--chain_type', 'tcrb',
            '--sequences', str(self.tcrb_chain_dataset_path),
            '--batch_size', '32',
            '--translate_to_asc'
        ]

        result = self.run_script('AlignAIRRPredict.py', args)
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")

        # Verify output file
        file_name = self.tcrb_chain_dataset_path.stem
        output_file = self.test_dir / f'{file_name}_alignairr_results.csv'
        self.assertTrue(output_file.exists(), "Output CSV not created")

        # Validate against expected results
        validation_file = self.test_dir / 'tcrb_predict_validation.csv'
        if validation_file.exists():
            self.assert_csv_equals(output_file, validation_file)

        # Cleanup
        if output_file.exists():
            output_file.unlink()

    def test_train_model_script(self):
        """Test model training script."""
        args = [
            '--chain_type', 'heavy',
            '--train_dataset', str(self.heavy_chain_dataset_path),
            '--session_path', './',
            '--epochs', '1',
            '--batch_size', '32',
            '--steps_per_epoch', '32',
            '--max_sequence_length', '576',
            '--model_name', 'TestModel'
        ]

        result = self.run_script('TrainModel.py', args)
        self.assertEqual(result.returncode, 0, f"Training script failed: {result.stderr}")

        # Verify model files
        model_dir = Path('./saved_models/TestModel')
        weights_file = model_dir / 'TestModel_weights_final_epoch.data-00000-of-00001'
        self.assertTrue(weights_file.exists(), "Model weights not created")


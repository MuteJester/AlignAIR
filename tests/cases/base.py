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


class BaseTestCase(unittest.TestCase):
    """Base test case with common setup and utilities."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources."""
        cls.test_dir = Path(__file__).parent.absolute()
        cls.src_dir = cls.test_dir.parent / 'src' / 'AlignAIR'
        cls.api_dir = cls.src_dir / 'API'

        # Test data paths
        cls.heavy_chain_dataset_path = cls.test_dir / 'sample_HeavyChain_dataset.csv'
        cls.light_chain_dataset_path = cls.test_dir / 'sample_LightChain_dataset.csv'
        cls.tcrb_chain_dataset_path = cls.test_dir / 'TCRB_Sample_Data.csv'

        # Model checkpoint paths
        cls.heavy_model_checkpoint = cls.test_dir / 'AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2'
        cls.light_model_checkpoint = cls.test_dir / 'LightChain_AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced'
        cls.tcrb_model_checkpoint = cls.test_dir / 'AlignAIRR_TCRB_Model_checkpoint'
        cls.genotyped_model_checkpoint = cls.test_dir / 'Genotyped_Frozen_Heavy_Chain_AlignAIRR_S5F_OGRDB_S5F_576_Balanced'

        # Python interpreter path
        cls.python_path = 'C:/Users/tomas/Desktop/AlignAIRR/AlignAIR_ENV/Scripts/python'

    def setUp(self):
        """Set up test-specific resources."""
        # Load test datasets
        if self.heavy_chain_dataset_path.exists():
            self.heavy_chain_dataset = pd.read_csv(self.heavy_chain_dataset_path)
        if self.light_chain_dataset_path.exists():
            self.light_chain_dataset = pd.read_csv(self.light_chain_dataset_path)

    def tearDown(self):
        """Clean up after each test."""
        # Clean up any temporary files created during tests
        temp_files = [
            'genotype.yaml',
            'model_params.yaml',
            'TestModel.csv'
        ]
        for file in temp_files:
            if Path(file).exists():
                Path(file).unlink()

        # Clean up directories
        temp_dirs = ['saved_models']
        for dir_name in temp_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                shutil.rmtree(dir_path)

    def run_script(self, script_name: str, args: List[str]) -> subprocess.CompletedProcess:
        """Run a script with given arguments and return the result."""
        script_path = self.api_dir / script_name
        self.assertTrue(script_path.exists(), f"Script not found: {script_path}")

        command = [self.python_path, str(script_path)] + args
        return subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

    def assert_csv_equals(self, actual_path: Path, expected_path: Path):
        """Compare two CSV files cell by cell."""
        actual_df = pd.read_csv(actual_path)
        expected_df = pd.read_csv(expected_path)

        self.assertEqual(actual_df.shape, expected_df.shape, "DataFrames have different shapes")

        for i in range(actual_df.shape[0]):
            for j in range(actual_df.shape[1]):
                self.assertEqual(
                    actual_df.iloc[i, j], expected_df.iloc[i, j],
                    f"Mismatch at row {i}, column {j}: {actual_df.iloc[i, j]} != {expected_df.iloc[i, j]}"
                )

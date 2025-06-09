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


class DataProcessingTests(BaseTestCase):
    """Test cases for data processing and feature extraction."""

    def test_fast_kmer_density_extractor(self):
        """Test k-mer density extraction for heavy chain sequences."""
        data_config_library = DataConfigLibrary()
        data_config_library.mount_type('heavy')

        ref_alleles = (
                data_config_library.reference_allele_sequences('v') +
                data_config_library.reference_allele_sequences('d') +
                data_config_library.reference_allele_sequences('j')
        )

        extractor = FastKmerDensityExtractor(11, max_length=576, allowed_mismatches=0)
        extractor.fit(ref_alleles)

        # Load test cases
        test_file = self.test_dir / 'KMer_Density_Extractor_HeavyChainTests.json'
        if test_file.exists():
            with open(test_file, 'r') as f:
                test_cases = json.load(f)

            results = [extractor.transform_holt(noise)[0] for true, noise in test_cases]

            for i, (result, (true, _)) in enumerate(zip(results, test_cases)):
                with self.subTest(test_case=i):
                    self.assertLessEqual(len(result), 576, "Result length exceeds maximum")
                    self.assertTrue(
                        self._is_within_mismatch_limit(true, result),
                        "True sequence not found within mismatch limit"
                    )

    def _is_within_mismatch_limit(self, true_seq: str, result_seq: str, max_mismatch: int = 5) -> bool:
        """Check if true sequence is contained in result with allowed mismatches."""
        true_len = len(true_seq)
        for i in range(max_mismatch + 1):
            if true_seq[i:] in result_seq or true_seq[:true_len - i] in result_seq:
                return True
        return False


class PreprocessingStepTests(BaseTestCase):
    """Test cases for preprocessing pipeline steps."""

    def test_config_load_step(self):
        """Test configuration loading step."""
        mock_predict_object = Mock()
        mock_predict_object.script_arguments.chain_type = 'heavy'
        mock_predict_object.script_arguments.heavy_data_config = 'path/to/heavy/config'
        mock_predict_object.script_arguments.kappa_data_config = 'path/to/kappa/config'
        mock_predict_object.script_arguments.lambda_data_config = 'path/to/lambda/config'

        mock_data_config_library = Mock(spec=DataConfigLibrary)
        mock_data_config_library.mount_type = Mock()

        with patch('src.AlignAIR.Preprocessing.Steps.dataconfig_steps.DataConfigLibrary',
                   return_value=mock_data_config_library):
            step = ConfigLoadStep("Load Config")
            result = step.process(mock_predict_object)

        mock_data_config_library.mount_type.assert_called_with('heavy')
        self.assertEqual(result.data_config_library, mock_data_config_library)
        self.assertTrue(mock_predict_object.mount_genotype_list.called)
        self.assertEqual(result, mock_predict_object)

    def test_file_name_extraction_step(self):
        """Test file name extraction step."""
        mock_predict_object = Mock()
        mock_predict_object.script_arguments.sequences = 'path/to/sequences/file.csv'

        step = FileNameExtractionStep("Extract File Name")
        result = step.process(mock_predict_object)

        self.assertIsInstance(result.file_info, FileInfo)
        self.assertEqual(result.file_info.file_name, 'file')
        self.assertEqual(result.file_info.file_type, 'csv')
        self.assertEqual(result, mock_predict_object)

    def test_file_sample_counter_step(self):
        """Test file sample counter step."""
        mock_predict_object = Mock()
        mock_predict_object.file_info.file_type = 'csv'
        mock_predict_object.script_arguments.sequences = str(self.heavy_chain_dataset_path)

        mock_row_counter = Mock(return_value=100)
        FILE_ROW_COUNTERS['csv'] = mock_row_counter

        step = FileSampleCounterStep("Count Samples")
        result = step.process(mock_predict_object)

        mock_row_counter.assert_called_with(str(self.heavy_chain_dataset_path))
        self.assertEqual(result, mock_predict_object)

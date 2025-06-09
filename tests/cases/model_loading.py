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


class ModelLoadingTests(BaseTestCase):
    """Test cases for model loading and inference."""

    def test_load_saved_heavy_chain_model(self):
        """Test loading a saved heavy chain model and making predictions."""
        model_params = {
            'max_seq_length': 576,
            'v_allele_count': 198,
            'd_allele_count': 34,
            'j_allele_count': 7
        }

        model = HeavyChainAlignAIRR(**model_params)
        trainer = Trainer(
            model=model,
            max_seq_length=model_params['max_seq_length'],
            epochs=1,
            batch_size=32,
            steps_per_epoch=1,
            verbose=1,
        )

        trainer.load_model(str(self.heavy_model_checkpoint))

        # Build model with dummy input
        dummy_input = {
            "tokenized_sequence": np.zeros((1, model_params['max_seq_length']), dtype=np.float32),
        }
        _ = trainer.model(dummy_input)

        # Test prediction
        prediction_dataset = PredictionDataset(max_sequence_length=576)
        test_seq = (
            'CAGCCACAACTGAACTGGTCAAGTCCAGGACTGGTGAATACCTCGCAGACCGTCACACTCACCCTTGCCGTGTCCGGGGACCGTGTCTCCAGAACCACTGCTGTTTGGAAGTGGAGGGGTCAGACCCCATCGCGAGGCCTTGCGTGGCTGGGAAGGACCTACNACAGTTCCAGGTGATTTGCTAACAACGAAGTGTCTGTGAATTGTTNAATATCCATGAACCCAGACGCATCCANGGAACGGNTCTTCCTGCACCTGAGGTCTGGGGCCTTCGACGACACGGCTGTACATNCGTGAGAAAGCGGTGACCTCTACTAGGATAGTGCTGAGTACGACTGGCATTACGCTCTCNGGGACCGTGCCACCCTTNTCACTGCCTCCTCGG')

        encoded_seq = prediction_dataset.encode_and_equal_pad_sequence(test_seq)['tokenized_sequence']
        predicted = trainer.model.predict({'tokenized_sequence': np.vstack([encoded_seq])})

        self.assertIsNotNone(predicted)


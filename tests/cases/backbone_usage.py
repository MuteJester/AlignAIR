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


class BackboneLoaderTests(BaseTestCase):
    """Test cases for backbone model loading with custom classification heads."""

    def test_heavy_chain_backbone_loader(self):
        """Test loading heavy chain backbone with custom classification heads."""
        from src.AlignAIR.Finetuning.CustomClassificationHeadLoader import CustomClassificationHeadLoader

        tf.get_logger().setLevel('ERROR')

        test_sizes = {
            'v_allele': 200,
            'd_allele': 20,
            'j_allele': 2
        }

        loader = CustomClassificationHeadLoader(
            pretrained_path=str(self.heavy_model_checkpoint),
            model_class=HeavyChainAlignAIRR,
            max_seq_length=576,
            pretrained_v_allele_head_size=198,
            pretrained_d_allele_head_size=34,
            pretrained_j_allele_head_size=7,
            custom_v_allele_head_size=test_sizes['v_allele'],
            custom_d_allele_head_size=test_sizes['d_allele'],
            custom_j_allele_head_size=test_sizes['j_allele']
        )

        # Verify custom head sizes
        for layer in loader.model.layers:
            for allele_type, expected_size in test_sizes.items():
                if layer.name == allele_type:
                    self.assertEqual(
                        layer.weights[0].shape[1], expected_size,
                        f"{allele_type} layer size mismatch"
                    )

        # Verify same number of layers
        self.assertEqual(len(loader.model.layers), len(loader.pretrained_model.layers))

        # Verify embedding weights are preserved
        self._verify_embedding_weights_preserved(loader.model, loader.pretrained_model)

    def _verify_embedding_weights_preserved(self, model1, model2):
        """Helper method to verify embedding weights are preserved between models."""
        for layer1, layer2 in zip(model1.layers, model2.layers):
            if 'embedding' in layer1.name and 'embedding' in layer2.name:
                weights1 = layer1.get_weights()
                weights2 = layer2.get_weights()

                for w1, w2 in zip(weights1, weights2):
                    try:
                        tf.debugging.assert_equal(w1, w2)
                    except tf.errors.InvalidArgumentError as e:
                        self.fail(f"Embedding weights mismatch in layer '{layer1.name}': {e}")


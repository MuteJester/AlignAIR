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


class OrientationClassifierTests(BaseTestCase):
    """Test cases for sequence orientation classification."""

    def test_heavy_chain_orientation_classifier(self):
        """Test orientation classification for heavy chain sequences."""
        classifier = builtin_orientation_classifier()
        self.assertIsNotNone(classifier)

        test_sequences = [
            'AGAACCCGTCCCTCAAGAGTCGGGTCACCNTATCAATAGACAAGTCCGAGAGCCAGTTCTCTCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTCTGTTCCTGTGCGAGATTCCATATGATAATTGAAAGTCTGCTTTGACCACTGGGACAAGGAAAACCTGGTCNCCGTCTCTTCAGNCTTCTACGACGGAC',
            'CGCCTNCGGACCCCTCTGCCACTGGGCCCAATGGACCGGGGTCAAGACCTTCACAACGTTATANGTAAGCAACGGTTCATGTGGTTAATTGCCGGCAAGGGAGCCAAAAGTCCGGGGGGTAANAGTGTATGCNGCACAAGAAAGTGAGTACAGACCTCTACCACTCAGACGGAAAGTGGCTGCGTCGTACAGGACAGCGCATTGACAATCAAAAAGAANATTATACCGATTAGGTGAGGTCGGGAAAGGGCCTTCCGACAGCCTGGGTCACGTGTCGTCTCAATTACTTCNACTCGGGTCTCCGACGTGTCCTCTCAAAGTCCCTGGGGGGTCCGGCCTGGTTCAGAGAGGGCCTGAGGTGGTTGCCGTGAAGTGGACGTGANTTTCCGGGAA',
            'CAAGACCGATAGGCCCTTGGTAGACGTCTGAGGGGACGGTGACCAGGGTGACCTGGNCCCAGTGCTGGAGGNTTGGGGATTCCATCGACGATCTNCACAAATTCGGGTACAATAAGACACGGCCGTGGCTGCGGTTCTCAGGCTGTTAANTTTTAAATACGCCGTGTGCTTTGAATCATTTCTGGAGATGGTGAATATGCCTTTCACCGAGATAGNATAATTCGTCGCACAAGTGTTAGATTTTTTTCTAGTATGGCCAANCCANTCCAGTTCTTTCCCGCAAGTCTGGCGGACCCACGGCAGAGAAGAGCCACTGAGGCTGATCTCAGAGCCTGCACCGCAGAGATTGAGGGAGCCACCAGGCTGGACCAAGTTTTTCCCGGACCCCACCAGGTGCATCTCGGCATGGCGTCTCAAGTTGACGTCNCTTGTGCCCGGTACCCTTTCTNTGGACCAACCGTAGGTGATCTCA',
            'AAGGTGCAGCTGGTAGAGTCTGGGGGAGGCTTGGTTCAGCCTGGAAGGTCCCTGAGACTCTCCTGTGCAGCCTCTTGATTCACCCTCAATAGCTATGCCATGCACCGGGTCCGCCAGGCTCCAGGCAANGGGCTGGAGTGGCTGGCANTTCTATCATATGATGGAAGTAATAGATCCTATGCAGACACCATGAAGGCCCGACTCACCATTTCCAGAGACAATTGCAANAACANGCTTTATATGCAAGTTAACAGCCTGNGACGTGAGGACACGGCTGTATATTAGTGTGCGAGGGGACTCCTAAGGTCCATAGCATCTGCTAGACTTTGACTACCGGGGCCAGGGAAACCTGGTCACCNTCTCCTCAGACTTCCACCAGGGGCCCATCGGTNTTCCC',
        ]

        expected_labels = ['Normal', 'Reversed', 'Reverse Complement', 'Normal']

        predicted_labels = classifier.predict(test_sequences[:4])  # Test subset for brevity

        for predicted, expected in zip(predicted_labels, expected_labels):
            self.assertEqual(predicted, expected)

    def test_light_chain_orientation_classifier(self):
        """Test orientation classification for light chain sequences."""
        classifier = builtin_orientation_classifier('light')

        test_sequences = [
            'CGCCTAGTTTGGTCAAGATATGTGCAATGTNAGCGGTCTGCCTCTGTAGCGTACTTGGTCAAGCTCTACTAGTAGGACAACCCAATTAGCTCCCCCCAGCCGTCTTCTGAGCTGCCTCNGGACCCTGCTGTGTCTGTGGCCTTNGGACAGACAGTCAGGATCACATGCNAAGGAGACAGCCTCAGAAGCTATTATGCAAGGTGGTCCCAGCAGAAGCCAGGGACAGGCCCCTGTACTTGTCATCNGTGGTAAAAACTACCGGCCCTCAAGGATCCCAGACCGATTCTCTGGCTCCAGCTCAGGAAACACAGCTTCCTTGGCCATCACTGGGGCTCAGGCGGATGATGAGGCTGACNATTACTGTANCTCCCGGGACAGCAGTGGTAACCNACTTTGTCTTCGGAACTGGGACCAAGGTCACCGTCCTAGGGTCAGCCCA',
            'TGGGAACAGAGTGACCGAGGGGGCAGGCTTGCGCTGACCCTCAGGGCGCTGACCTTGGTCCCAGTTCCTANGATTAATGCCCGTACCCCAGGTCTGAGAGTAATAGTCAGCCTTCATCCTCAAACTGGAGGCTGGAGATGGNGATGGCNGTTCTCAGCCCCAGAGCTNGNGTCTGAAGAGCGATCAGGGATCCCGTCCCTCTTGCTGTGACTGCCCTCACTGTTAAGCGTCATCAAGTAGCGAGGCCCCCTCTCTGCTGCTGTTGATGTCATGTGACGGCGTAGCCGGT',
        ]

        expected_labels = ['Normal', 'Reverse Complement']

        predicted_labels = classifier.predict(test_sequences)

        for predicted, expected in zip(predicted_labels, expected_labels):
            self.assertEqual(predicted, expected)


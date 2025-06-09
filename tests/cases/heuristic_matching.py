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


class HeuristicMatchingTests(BaseTestCase):
    """Test cases for heuristic reference matching."""

    def setUp(self):
        super().setUp()
        self.ighv_dc = builtin_heavy_chain_data_config()
        self.ighv_reference = {
            allele.name: allele.ungapped_seq.upper()
            for gene_family in self.ighv_dc.v_alleles
            for allele in self.ighv_dc.v_alleles[gene_family]
        }
        self.matcher = HeuristicReferenceMatcher(self.ighv_reference)

    def test_heuristic_matcher_basic(self):
        """Test basic heuristic matching functionality."""
        sequences = [
            'ATCGCAGGTCACCTTGAAGGAGTCTGGTCCTGTGCTGGTGAAACCCACAGAGACCCTCACGCTGACCTGCACCGTCTCTGGGTTCTCACTCAGCAATGCTAGAATGGGTGTGAGCTGGATCCGTCAGCCCCCAGGGAAGGCCCTGGAGTGGCTTGCACACATTTTTTCGAATGACGAAAAATCCTACAGCACATCTCTGAAGAGCAGGCTCACCATCTCCAAGGACACCTCCAAAAGCCAGGTGGTCCTTACCATGACCAATATGGACCCTGTGGACACAGCCACATATTACTGTGCATGGATACATCG',
            'ATCGCGCAGGTCACCTTGAAGGAGTCTGGTCCTGTGCTGGTGAAACCCACAGAGACCCTCACGCTGACCTGCACCGTCTCTGGGTTCTCACTCAGCAATGCTAGAATGGGTGTGAGCTGGATCCGTCAGCCCCCAGGGAAGGCCCTGGAGTGGCTTGCACACATTTTTTCGAATGACGAAAAATCCTACAGCACATCTCTGAAGAGCAGGCTCACCATCTCCAAGGACACCTCCAAAAGCCAGGTGGTCCTTACCATGACCAATATGGACCCTGTGGACACAGCCACATATTACTGTGCATGGATACATCG',
        ]
        starts = [3, 5]
        ends = [303, 305]
        indel_counts = [0, 0]
        alleles = ['IGHVF1-G1*01', 'IGHVF1-G1*01']

        results = self.matcher.match(sequences, starts, ends, alleles, indel_counts=indel_counts, _gene='v')

        expected_results = [
            {'start_in_ref': 0, 'end_in_ref': 301},
            {'start_in_ref': 0, 'end_in_ref': 301}
        ]

        for i, (result, expected) in enumerate(zip(results, expected_results)):
            with self.subTest(sequence=i):
                self.assertEqual(result['start_in_ref'], expected['start_in_ref'])
                self.assertEqual(result['end_in_ref'], expected['end_in_ref'])

    def test_heuristic_matcher_comprehensive(self):
        """Test comprehensive heuristic matching with various edge cases."""
        ref_seq = "ATGCGTACGTCAGTACGTCAGTACGTTAGC"  # length = 30
        allele = "TEST*01"
        matcher = HeuristicReferenceMatcher({allele: ref_seq})

        def mutate(seq, pos, base):
            return seq[:pos] + base + seq[pos + 1:]

        test_cases = [
            ("exact", lambda: ref_seq, 0, 0, 30),
            ("overhang_left", lambda: "GGG" + ref_seq, 0, 0, 30),
            ("overhang_right", lambda: ref_seq + "TTTT", 0, 0, 30),
            ("overhang_both", lambda: "AA" + ref_seq + "CC", 0, 0, 30),
            ("del_left", lambda: ref_seq[5:], 5, 5, 30),
            ("del_right", lambda: ref_seq[:-6], 6, 0, 24),
            ("del_both", lambda: ref_seq[3:-4], 7, 3, 26),
            ("internal_insert", lambda: ref_seq[:10] + "NNNN" + ref_seq[10:], 4, 0, 30),
            ("two_mismatches", lambda: mutate(mutate(ref_seq, 5, "A"), 15, "C"), 0, 0, 30),
            ("mixed_combo", lambda: "GG" + ref_seq[:12] + "NN" + mutate(ref_seq[12:], 5, "G"), 2, 0, 30),
        ]

        sequences, starts, ends, alleles, indel_counts = [], [], [], [], []
        expected_results = []

        for name, builder, indels, exp_start, exp_end in test_cases:
            seg = builder()
            sequences.append(seg)
            starts.append(0)
            ends.append(len(seg))
            alleles.append(allele)
            indel_counts.append(indels)
            expected_results.append((name, exp_start, exp_end))

        results = matcher.match(sequences, starts, ends, alleles, indel_counts, _gene="v")

        for (name, exp_start, exp_end), result in zip(expected_results, results):
            with self.subTest(case=name):
                self.assertEqual(result["start_in_ref"], exp_start, f"{name}: start mismatch")
                self.assertEqual(result["end_in_ref"], exp_end, f"{name}: end mismatch")


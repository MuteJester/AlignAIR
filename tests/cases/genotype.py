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


class GenotypeTests(BaseTestCase):
    """Test cases for genotype correction functionality."""

    def test_genotype_correction(self):
        """Test genotype correction workflow."""
        # Load genotype data config
        genotype_config_file = self.test_dir / 'Genotyped_DataConfig.pkl'
        if not genotype_config_file.exists():
            self.skipTest("Genotyped_DataConfig.pkl not found")

        with open(genotype_config_file, 'rb') as f:
            genotype_dataconfig = pickle.load(f)

        ref_dataconfig = builtin_heavy_chain_data_config()

        # Extract allele names
        v_alleles = [allele.name for gene_family in genotype_dataconfig.v_alleles
                     for allele in genotype_dataconfig.v_alleles[gene_family]]
        d_alleles = [allele.name for gene_family in genotype_dataconfig.d_alleles
                     for allele in genotype_dataconfig.d_alleles[gene_family]] + ['Short-D']
        j_alleles = [allele.name for gene_family in genotype_dataconfig.j_alleles
                     for allele in genotype_dataconfig.j_alleles[gene_family]]

        v_alleles_ref = [allele.name for gene_family in ref_dataconfig.v_alleles
                         for allele in ref_dataconfig.v_alleles[gene_family]]
        d_alleles_ref = [allele.name for gene_family in ref_dataconfig.d_alleles
                         for allele in ref_dataconfig.d_alleles[gene_family]] + ['Short-D']
        j_alleles_ref = [allele.name for gene_family in ref_dataconfig.j_alleles
                         for allele in ref_dataconfig.j_alleles[gene_family]]

        # Create temporary configuration files
        genotype_file = Path('genotype.yaml')
        model_params_file = Path('model_params.yaml')

        try:
            # Create genotype configuration
            with open(genotype_file, 'w') as f:
                yaml.dump({'v': v_alleles, 'd': d_alleles, 'j': j_alleles}, f)

            # Create model parameters
            mock_model_params = {
                'v_allele_latent_size': 2 * len(v_alleles_ref),
                'd_allele_latent_size': 2 * len(d_alleles_ref),
                'j_allele_latent_size': 2 * len(j_alleles_ref),
                'v_allele_count': len(v_alleles),
                'd_allele_count': len(d_alleles),
                'j_allele_count': len(j_alleles)
            }

            with open(model_params_file, 'w') as f:
                yaml.dump(mock_model_params, f)

            # Test full model prediction
            args_full = [
                '--model_checkpoint', str(self.heavy_model_checkpoint),
                '--save_path', f'{self.test_dir}/',
                '--chain_type', 'heavy',
                '--sequences', str(self.heavy_chain_dataset_path),
                '--batch_size', '32',
                '--translate_to_asc',
                '--save_predict_object',
                '--custom_genotype', str(genotype_file)
            ]

            result = self.run_script('AlignAIRRPredict.py', args_full)
            self.assertEqual(result.returncode, 0, f"Full model script failed: {result.stderr}")

            # Load full model predictions
            file_name = self.heavy_chain_dataset_path.stem
            predict_object_path = self.test_dir / f'{file_name}_alignair_results_predictObject.pkl'

            if predict_object_path.exists():
                with open(predict_object_path, 'rb') as f:
                    predict_object_full = pickle.load(f)

                # Test genotyped model prediction
                if self.genotyped_model_checkpoint.exists():
                    args_genotyped = [
                        '--model_checkpoint', str(self.genotyped_model_checkpoint),
                        '--save_path', f'{self.test_dir}/',
                        '--chain_type', 'heavy',
                        '--sequences', str(self.heavy_chain_dataset_path),
                        '--heavy_data_config', str(genotype_config_file),
                        '--batch_size', '32',
                        '--translate_to_asc',
                        '--save_predict_object',
                        '--finetuned_model_params_yaml', str(model_params_file),
                        '--custom_genotype', str(genotype_file)
                    ]

                    result = self.run_script('AlignAIRRPredict.py', args_genotyped)
                    self.assertEqual(result.returncode, 0, f"Genotyped model script failed: {result.stderr}")

                    # Load genotyped model predictions
                    with open(predict_object_path, 'rb') as f:
                        predict_object_genotyped = pickle.load(f)

                    # Compare predictions
                    for allele_type in ['v_allele', 'd_allele', 'j_allele']:
                        if (allele_type in predict_object_full.processed_predictions and
                                allele_type in predict_object_genotyped.processed_predictions):
                            mae = np.mean(np.abs(
                                predict_object_full.processed_predictions[allele_type] -
                                predict_object_genotyped.processed_predictions[allele_type]
                            ))

                            # Assert reasonable similarity between predictions
                            self.assertLess(mae, 1.0, f"High MAE for {allele_type}: {mae}")

        finally:
            # Cleanup temporary files
            for temp_file in [genotype_file, model_params_file]:
                if temp_file.exists():
                    temp_file.unlink()

            # Cleanup output files
            output_files = [
                self.test_dir / f'{self.heavy_chain_dataset_path.stem}_alignairr_results.csv',
                self.test_dir / f'{self.heavy_chain_dataset_path.stem}_alignair_results_predictObject.pkl'
            ]
            for output_file in output_files:
                if output_file.exists():
                    output_file.unlink()


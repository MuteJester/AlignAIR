import os
import sys
import unittest
import pandas as pd
import numpy as np
import subprocess
import re
from pathlib import Path
import tempfile
import pytest
import importlib.util

pytestmark = [pytest.mark.e2e]


class TestModule(unittest.TestCase):
    def setUp(self):
        repo_root = Path(__file__).resolve().parents[2]
        self.repo_root = repo_root
        self.data_test_dir = repo_root / 'tests' / 'data' / 'test'
        self.data_val_dir = repo_root / 'tests' / 'data' / 'validation'
        self.checkpoints_dir = repo_root / 'checkpoints'

        self.heavy_chain_dataset_path = str(self.data_test_dir / 'sample_igh.csv')
        self.light_chain_dataset_path = str(self.data_test_dir / 'sample_igl_k.csv')
        self.tcrb_chain_dataset_path = str(self.data_test_dir / 'sample_tcrb.csv')

        # Ensure datasets are readable
        pd.read_csv(self.heavy_chain_dataset_path)
        pd.read_csv(self.light_chain_dataset_path)
        # Optional dependency used by AlignAIRRPredict CLI
        self._has_questionary = importlib.util.find_spec('questionary') is not None

    def _with_env(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.repo_root / 'src') + (
            os.pathsep + env['PYTHONPATH'] if 'PYTHONPATH' in env and env['PYTHONPATH'] else ''
        )
        return env

    def test_predict_script_single_chain(self):
        if not self._has_questionary:
            self.skipTest('questionary package not installed')
        ckpt_prefix = self.checkpoints_dir / 'IGH_S5F_576_EXTENDED'
        if not ((ckpt_prefix.parent / f"{ckpt_prefix.name}.index").exists()):
            self.skipTest('IGH_S5F_576_EXTENDED checkpoint not available')

        sequences_path = str(self.data_test_dir / 'sample_igh_extended.csv')
        validation_path = str(self.data_val_dir / 'igh_model_prediction_validation.csv')

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = tmpdir + os.sep
            cmd = [
                sys.executable, '-m', 'AlignAIR.API.AlignAIRRPredict',
                '--model_checkpoint', str(ckpt_prefix),
                '--save_path', save_dir,
                '--genairr_dataconfig', 'HUMAN_IGH_EXTENDED',
                '--sequences', sequences_path,
                '--batch_size', '16',
                '--translate_to_asc',
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=self._with_env())
            if res.returncode != 0:
                print('STDOUT:', res.stdout)
                print('STDERR:', res.stderr)
            self.assertEqual(res.returncode, 0, 'AlignAIRRPredict CLI failed')

            out_csv = Path(tmpdir) / f"{Path(sequences_path).stem}_alignairr_results.csv"
            self.assertTrue(out_csv.is_file(), 'Expected output CSV not found')

            df = pd.read_csv(out_csv)
            validation = pd.read_csv(validation_path)

            self.assertTrue((df['sequence'] == validation['sequence']).all())
            for gene in ['v', 'd', 'j']:
                for pos in ['start', 'end']:
                    for loc in ['sequence', 'germline']:
                        col = f'{gene}_{loc}_{pos}'
                        self.assertTrue((df[col] == validation[col]).all(), f'mismatch in {col}')
            for gene in ['v', 'd', 'j']:
                col = f'{gene}_call'
                self.assertTrue((df[col] == validation[col]).all(), f'mismatch in {col}')
            for gene in ['v', 'd', 'j']:
                col = f'{gene}_likelihoods'
                validation[col] = validation[col].apply(lambda x: re.findall(r"[-+]?\d*\.\d+|\d+", x))
                validation[col] = validation[col].apply(lambda x: [float(i) for i in x])
                df[col] = df[col].apply(lambda x: re.findall(r"[-+]?\d*\.\d+|\d+", x))
                df[col] = df[col].apply(lambda x: [float(i) for i in x])
                df[col] = df[col].apply(lambda x: [round(i, 3) for i in x])
                validation[col] = validation[col].apply(lambda x: [round(i, 3) for i in x])
                self.assertTrue((df[col] == validation[col]).all(), f'mismatch in {col}')
            self.assertTrue((df['productive'] == validation['productive']).all(), 'productive mismatch')
            self.assertTrue(np.allclose(df['indels'], validation['indels'], atol=1e-3))
            self.assertTrue(np.allclose(df['mutation_rate'], validation['mutation_rate'], atol=1e-3))

    def test_predict_script_tcrb(self):
        if not self._has_questionary:
            self.skipTest('questionary package not installed')
        ckpt_prefix = self.checkpoints_dir / 'TCRB_UNIFORM_576'
        if not ((ckpt_prefix.parent / f"{ckpt_prefix.name}.index").exists()):
            self.skipTest('TCRB_UNIFORM_576 checkpoint not available')

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = tmpdir + os.sep
            cmd = [
                sys.executable, '-m', 'AlignAIR.API.AlignAIRRPredict',
                '--model_checkpoint', str(ckpt_prefix),
                '--save_path', save_dir,
                '--genairr_dataconfig', 'HUMAN_TCRB_IMGT',
                '--sequences', self.tcrb_chain_dataset_path,
                '--batch_size', '16',
                '--translate_to_asc',
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=self._with_env())
            if res.returncode != 0:
                print('STDOUT:', res.stdout)
                print('STDERR:', res.stderr)
            self.assertEqual(res.returncode, 0, 'AlignAIRRPredict CLI failed for TCRB')

            out_csv = Path(tmpdir) / f"{Path(self.tcrb_chain_dataset_path).stem}_alignairr_results.csv"
            self.assertTrue(out_csv.is_file(), 'Expected output CSV not found')
            df = pd.read_csv(out_csv)
            validation = pd.read_csv(str(self.data_val_dir / 'tcrb_model_prediction_validation.csv'))
            if 'chain_type' in df.columns:
                df = df.drop(columns=['chain_type'])
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    self.assertEqual(df.iloc[i, j], validation.iloc[i, j])
            self.assertFalse(df.empty)

    @pytest.mark.e2e
    def test_train_model_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable, '-m', 'AlignAIR.API.TrainModel',
                '--genairr_dataconfigs', 'HUMAN_IGH_OGRDB',
                '--train_datasets', self.heavy_chain_dataset_path,
                '--session_path', tmpdir,
                '--epochs', '1',
                '--batch_size', '8',
                '--samples_per_epoch', '16',
                '--max_sequence_length', '576',
                '--model_name', 'TestModel'
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=self._with_env())
            if res.returncode != 0:
                print('STDOUT:', res.stdout)
                print('STDERR:', res.stderr)
            self.assertEqual(res.returncode, 0)
            self.assertTrue((Path(tmpdir) / 'TestModel_final.weights.h5').is_file())

    @pytest.mark.e2e
    def test_train_model_script_multi_chain_cli(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable, '-m', 'AlignAIR.API.TrainModel',
                '--genairr_dataconfigs', 'HUMAN_IGH_OGRDB', 'HUMAN_IGK_OGRDB',
                '--train_datasets', self.heavy_chain_dataset_path, self.light_chain_dataset_path,
                '--session_path', tmpdir,
                '--epochs', '1',
                '--batch_size', '8',
                '--samples_per_epoch', '16',
                '--max_sequence_length', '576',
                '--model_name', 'TestModelMultiCLI'
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=self._with_env())
            if res.returncode != 0:
                print('STDOUT:', res.stdout)
                print('STDERR:', res.stderr)
            self.assertEqual(res.returncode, 0)
            self.assertTrue((Path(str(tmpdir)+'/TestModelMultiCLI_final.weights.h5')).is_file())

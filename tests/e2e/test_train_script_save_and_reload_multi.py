import os
import sys
import unittest
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import pytest

pytestmark = [pytest.mark.e2e]


class TestTrainScriptSaveAndReloadMulti(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[2]
        self.data_test_dir = self.repo_root / 'tests' / 'data' / 'test'
        # Use the same light chain file twice to keep CI simple and fast
        self.light_chain_dataset_path = str(self.data_test_dir / 'sample_igl_k.csv')

        self.env = os.environ.copy()
        self.env['PYTHONPATH'] = str(self.repo_root / 'src') + (
            os.pathsep + self.env['PYTHONPATH'] if 'PYTHONPATH' in self.env and self.env['PYTHONPATH'] else ''
        )
        self.env['WANDB_MODE'] = 'offline'

        if str(self.repo_root / 'src') not in sys.path:
            sys.path.insert(0, str(self.repo_root / 'src'))

    def test_single_epoch_train_and_reload_multi_chain(self):
        script_path = self.repo_root / 'jenkins_scripts' / 'AlignAIR_Train.py'
        if not script_path.is_file():
            self.skipTest('AlignAIR_Train.py script not found')

        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = 'TestScriptModelMulti'
            cmd = [
                sys.executable,
                str(script_path),
                '--genairr_dataconfigs', 'HUMAN_IGK_OGRDB', 'HUMAN_IGL_OGRDB',
                '--train_datasets', self.light_chain_dataset_path, self.light_chain_dataset_path,
                '--session_path', tmpdir,
                '--epochs', '1',
                '--batch_size', '8',
                '--samples_per_epoch', '16',
                '--max_sequence_length', '576',
                '--model_name', model_name,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=self.env)
            if res.returncode != 0:
                print('STDOUT:', res.stdout)
                print('STDERR:', res.stderr)
            self.assertEqual(res.returncode, 0, 'Train script failed for multi-chain')

            bundle_dir = Path(tmpdir) / model_name
            self.assertTrue((bundle_dir / 'config.json').is_file(), 'config.json missing in bundle')
            self.assertTrue((bundle_dir / 'dataconfig.pkl').is_file(), 'dataconfig.pkl missing in bundle')
            self.assertTrue((bundle_dir / 'weights.h5').is_file(), 'weights.h5 missing in bundle')

            from AlignAIR.Models.MultiChainAlignAIR.MultiChainAlignAIR import MultiChainAlignAIR
            model = MultiChainAlignAIR.from_pretrained(str(bundle_dir))
            L = int(getattr(model, 'max_seq_length', 576))
            dummy = {"tokenized_sequence": np.zeros((1, L), dtype=np.int32)}
            out = model(dummy, training=False)
            self.assertIn('chain_type', out)
            # Expect the number of types equals the config's chain_types length
            self.assertGreater(out['chain_type'].shape[-1], 1, 'Chain type logits should have >1 class')

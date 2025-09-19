import os
import unittest
import numpy as np
from pathlib import Path
import pytest

from AlignAIR.Data import MultiDataConfigContainer
from AlignAIR.Models.SingleChainAlignAIR.SingleChainAlignAIR import SingleChainAlignAIR
from AlignAIR.Data.PredictionDataset import PredictionDataset
from GenAIRR.data import _CONFIG_NAMES
from GenAIRR import data as _gen_data

for _cfg in _CONFIG_NAMES:
	globals()[_cfg] = getattr(_gen_data, _cfg)

from GenAIRR.data import HUMAN_IGH_OGRDB

# Mark this module as an integration test suite
pytestmark = pytest.mark.integration

# Resolve checkpoint prefix robustly from repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_PREFIX = REPO_ROOT / "checkpoints" / "IGH_S5F_576"
_index_exists = (CKPT_PREFIX.parent / f"{CKPT_PREFIX.name}.index").exists()
_data_exists = any(CKPT_PREFIX.parent.glob(f"{CKPT_PREFIX.name}.data-*") )
_CKPT_AVAILABLE = _index_exists and _data_exists


class TestModule(unittest.TestCase):
	@pytest.mark.skipif(not _CKPT_AVAILABLE, reason="IGH_S5F_576 checkpoint files not found under ./checkpoints; skipping integration test.")
	def test_load_saved_single_model(self):
		model_params = {
			'max_seq_length': 576,
			'dataconfig': MultiDataConfigContainer([HUMAN_IGH_OGRDB]),
		}
		model = SingleChainAlignAIR(**model_params)
		dummy_input = {"tokenized_sequence": np.zeros((1, 576), dtype=np.float32)}
		_ = model(dummy_input)
		model_checkpoint_path = str(CKPT_PREFIX)
		model.load_weights(model_checkpoint_path).expect_partial()
		prediction_dataset = PredictionDataset(max_sequence_length=576)
		seq = 'CAGCCACAACTGAACTGGTCAAGTCCAGGACTGGTGAATACCTCGCAGACCGTCACACTCACCCTTGCCGTGTCCGGGGACCGTGTCTCCAGAACCACTGCTGTTTGGAAGTGGAGGGGTCAGACCCCATCGCGAGGCCTTGCGTGGCTGGGAAGGACCTACNACAGTTCCAGGTGATTTGCTAACAACGAAGTGTCTGTGAATTGTTNAATATCCATGAACCCAGACGCATCCANGGAACGGNTCTTCCTGCACCTGAGGTCTGGGGCCTTCGACGACACGGCTGTACATNCGTGAGAAAGCGGTGACCTCTACTAGGATAGTGCTGAGTACGACTGGCATTACGCTCTCNGGGACCGTGCCACCCTTNTCACTGCCTCCTCGG'
		encoded_seq = prediction_dataset.encode_and_equal_pad_sequence(seq)['tokenized_sequence']
		predicted = model.predict({'tokenized_sequence': np.vstack([encoded_seq])})
		self.assertIsNotNone(predicted, "Prediction should not be None after loading weights.")
		self.assertIn('v_allele', predicted)


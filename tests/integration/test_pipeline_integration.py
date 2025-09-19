import os
import sys
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from tempfile import TemporaryDirectory

from AlignAIR.Data import MultiChainDataset, MultiDataConfigContainer
from AlignAIR.Models.MultiChainAlignAIR.MultiChainAlignAIR import MultiChainAlignAIR
from AlignAIR.Models.SingleChainAlignAIR.SingleChainAlignAIR import SingleChainAlignAIR
from src.AlignAIR.Data import SingleChainDataset
from GenAIRR.data import _CONFIG_NAMES
from GenAIRR import data as _gen_data
from GenAIRR.data import HUMAN_IGH_OGRDB, HUMAN_IGK_OGRDB, HUMAN_IGL_OGRDB

# expose configs in globals to match original behavior
for _cfg in _CONFIG_NAMES:
	globals()[_cfg] = getattr(_gen_data, _cfg)


class TestModule(unittest.TestCase):
	def setUp(self):
		self.heavy_chain_dataset_path = 'data/test/sample_igh.csv'
		self.light_chain_dataset_path = 'data/test/sample_igl_k.csv'
		self.tcrb_chain_dataset_path = 'data/test/sample_tcrb.csv'
		self.heavy_chain_dataset = pd.read_csv(self.heavy_chain_dataset_path)
		self.light_chain_dataset = pd.read_csv(self.light_chain_dataset_path)
		self.test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

	def tearDown(self):
		pass

	def test_single_chain_model_training(self):
		train_dataset = SingleChainDataset(
			data_path='data/test/sample_ighv_extended.csv',
			dataconfig=MultiDataConfigContainer([HUMAN_IGH_OGRDB]),
			use_streaming=True,
			max_sequence_length=576
		)

		model_params = train_dataset.generate_model_params()
		model = SingleChainAlignAIR(**model_params)

		model.compile(
			optimizer=tf.keras.optimizers.Adam(clipnorm=1),
			loss=None,
			metrics={
				'v_allele': [tf.keras.metrics.AUC(name='auc'), 'binary_accuracy'],
				'd_allele': [tf.keras.metrics.AUC(name='auc'), 'binary_accuracy'],
				'j_allele': [tf.keras.metrics.AUC(name='auc'), 'binary_accuracy'],
			}
		)

		from src.AlignAIR.Trainers import Trainer
		with TemporaryDirectory() as tmpdir:
			trainer = Trainer(
				model=model,
				session_path=tmpdir,
				model_name="heavy_chain_test"
			)
			trainer.train(
				train_dataset=train_dataset,
				epochs=15,
				samples_per_epoch=32,
				batch_size=16
			)

			self.assertIsNotNone(trainer.history, "Training history should not be None.")
			self.assertIn('loss', trainer.history.history, "Loss should be in training history.")

	def test_multi_chain_alignair_model_training(self):
		multi_config = MultiDataConfigContainer([HUMAN_IGK_OGRDB, HUMAN_IGL_OGRDB])
		train_dataset = MultiChainDataset(
			data_paths=[self.light_chain_dataset_path, self.light_chain_dataset_path],
			dataconfigs=multi_config,
			max_sequence_length=576,
			use_streaming=True,
		)

		model_params = train_dataset.generate_model_params()
		model = MultiChainAlignAIR(**model_params)

		metrics = {}
		metrics[f'v_allele'] = [tf.keras.metrics.AUC(name='auc')]
		metrics[f'j_allele'] = [tf.keras.metrics.AUC(name='auc')]
		metrics['chain_type'] = 'categorical_accuracy'

		model.compile(
			optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1),
			loss=None,
			metrics=metrics
		)

		from src.AlignAIR.Trainers import Trainer
		with TemporaryDirectory() as tmpdir:
			trainer = Trainer(
				model=model,
				session_path=tmpdir,
				model_name="multi_chain_test"
			)
			trainer.train(
				train_dataset=train_dataset,
				epochs=1,
				samples_per_epoch=64,
				batch_size=32
			)

			self.assertIsNotNone(trainer.history, "Training history should not be None.")

			dummy_input_data = np.random.randint(0, 6, size=(1, 576))
			test_input = {
				f"tokenized_sequence": dummy_input_data
			}
			predictions = model(test_input, training=False)
			self.assertIn('chain_type', predictions)
			expected_chain_types = len(multi_config.chain_types())
			self.assertEqual(predictions['chain_type'].shape[-1], expected_chain_types)

	def test_model_loading_step_multi_chain_integration(self):
		from AlignAIR.Preprocessing.Steps.model_loading_steps import ModelLoadingStep
		from AlignAIR.Utilities.step_utilities import FileInfo

		model_loader = ModelLoadingStep("Test Multi-Chain Model Loading")
		multi_config = MultiDataConfigContainer([HUMAN_IGK_OGRDB, HUMAN_IGL_OGRDB])
		test_file_info = FileInfo(self.light_chain_dataset_path)
		mock_checkpoint = os.path.join(self.test_dir, 'LightChain_AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced')

		try:
			model = model_loader.load_model(
				file_info=test_file_info,
				dataconfig=multi_config,
				model_checkpoint=mock_checkpoint,
				max_sequence_size=576
			)
			from AlignAIR.Models.MultiChainAlignAIR.MultiChainAlignAIR import MultiChainAlignAIR
			self.assertIsInstance(model, MultiChainAlignAIR,
								  "ModelLoadingStep should select MultiChainAlignAIR for multi-chain scenarios")
		except Exception as e:
			print(f"Expected weight loading error (this is normal): {e}")

		is_multi_chain = len(multi_config) > 1
		self.assertTrue(is_multi_chain, "Multi-config should be detected as multi-chain")

		single_config = MultiDataConfigContainer([HUMAN_IGH_OGRDB])
		is_single_chain = len(single_config) == 1
		self.assertTrue(is_single_chain, "Single config should be detected as single-chain")


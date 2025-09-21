import os
import sys
import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
import pytest

# Ensure local 'src' is importable when running this file directly
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from GenAIRR.data import _CONFIG_NAMES
from GenAIRR import data as _gen_data
for _cfg in _CONFIG_NAMES:
	globals()[_cfg] = getattr(_gen_data, _cfg)

from GenAIRR.data import HUMAN_IGH_OGRDB, HUMAN_IGK_OGRDB, HUMAN_IGL_OGRDB
from AlignAIR.Data import MultiDataConfigContainer, MultiChainDataset, SingleChainDataset
from AlignAIR.Models.SingleChainAlignAIR.SingleChainAlignAIR import SingleChainAlignAIR
from AlignAIR.Models.MultiChainAlignAIR.MultiChainAlignAIR import MultiChainAlignAIR


class TestModule(unittest.TestCase):
	def setUp(self):
		# Resolve dataset CSVs from repo fixtures (prefer tests/data/test, fallback to data/test)
		def _resolve_dataset(filename: str) -> Path:
			candidates = [
				REPO_ROOT / 'tests' / 'data' / 'test' / filename,
				REPO_ROOT / 'data' / 'test' / filename,
			]
			for p in candidates:
				if p.is_file():
					return p
			self.skipTest(f"Missing dataset fixture for {filename}; looked in {candidates}")

		self.heavy_chain_dataset_path = str(_resolve_dataset('sample_igh.csv'))
		self.light_chain_dataset_path = str(_resolve_dataset('sample_igl_k.csv'))
		self.tcrb_chain_dataset_path = str(_resolve_dataset('sample_tcrb.csv'))
		self.test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

	def tearDown(self):
		pass

	@pytest.mark.integration
	def test_heavy_chain_backbone_loader(self):
		from AlignAIR.Finetuning.CustomClassificationHeadLoader import CustomClassificationHeadLoader
		from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
		tf.get_logger().setLevel('ERROR')

		TEST_V_ALLELE_SIZE = 200
		TEST_D_ALLELE_SIZE = 20
		TEST_J_ALLELE_SIZE = 2

		# Pretrained path must exist locally; otherwise skip this environment-dependent test
		pretrained_dir = REPO_ROOT / 'AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2'
		if not pretrained_dir.exists():
			self.skipTest(f"Pretrained backbone not available at {pretrained_dir}")

		loader = CustomClassificationHeadLoader(
			pretrained_path=str(pretrained_dir),
			model_class=HeavyChainAlignAIRR,
			max_seq_length=576,
			pretrained_v_allele_head_size=198,
			pretrained_d_allele_head_size=34,
			pretrained_j_allele_head_size=7,
			custom_v_allele_head_size=TEST_V_ALLELE_SIZE,
			custom_d_allele_head_size=TEST_D_ALLELE_SIZE,
			custom_j_allele_head_size=TEST_J_ALLELE_SIZE,
		)

		for layer in loader.model.layers:
			if 'v_allele' == layer.name:
				self.assertEqual(layer.weights[0].shape[1], TEST_V_ALLELE_SIZE)
			if 'd_allele' == layer.name:
				self.assertEqual(layer.weights[0].shape[1], TEST_D_ALLELE_SIZE)
			if 'j_allele' == layer.name:
				self.assertEqual(layer.weights[0].shape[1], TEST_J_ALLELE_SIZE)

		self.assertEqual(len(loader.model.layers), len(loader.pretrained_model.layers))

		for layer, pretrained_layer in zip(loader.model.layers, loader.pretrained_model.layers):
			if 'embedding' in layer.name and 'embedding' in pretrained_layer.name:
				weights_model1 = layer.get_weights()
				weights_model2 = pretrained_layer.get_weights()
				for w1, w2 in zip(weights_model1, weights_model2):
					try:
						tf.debugging.assert_equal(w1, w2)
					except tf.errors.InvalidArgumentError as e:
						self.fail(f"Weights mismatch in embedding layer '{layer.name}': {e}")

	@pytest.mark.integration
	def test_light_chain_backbone_loader(self):
		from AlignAIR.Finetuning.CustomClassificationHeadLoader import CustomClassificationHeadLoader
		from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
		tf.get_logger().setLevel('ERROR')

		TEST_V_ALLELE_SIZE = 200
		TEST_D_ALLELE_SIZE = 20
		TEST_J_ALLELE_SIZE = 2

		# Pretrained path must exist locally; otherwise skip this environment-dependent test
		pretrained_dir = REPO_ROOT / 'AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2'
		if not pretrained_dir.exists():
			self.skipTest(f"Pretrained backbone not available at {pretrained_dir}")

		loader = CustomClassificationHeadLoader(
			pretrained_path=str(pretrained_dir),
			model_class=HeavyChainAlignAIRR,
			max_seq_length=576,
			pretrained_v_allele_head_size=198,
			pretrained_d_allele_head_size=34,
			pretrained_j_allele_head_size=7,
			custom_v_allele_head_size=TEST_V_ALLELE_SIZE,
			custom_d_allele_head_size=TEST_D_ALLELE_SIZE,
			custom_j_allele_head_size=TEST_J_ALLELE_SIZE,
		)

		for layer in loader.model.layers:
			if 'v_allele' == layer.name:
				self.assertEqual(layer.weights[0].shape[1], TEST_V_ALLELE_SIZE)
			if 'd_allele' == layer.name:
				self.assertEqual(layer.weights[0].shape[1], TEST_D_ALLELE_SIZE)
			if 'j_allele' == layer.name:
				self.assertEqual(layer.weights[0].shape[1], TEST_J_ALLELE_SIZE)

		self.assertEqual(len(loader.model.layers), len(loader.pretrained_model.layers))

		for layer, pretrained_layer in zip(loader.model.layers, loader.pretrained_model.layers):
			if 'embedding' in layer.name and 'embedding' in pretrained_layer.name:
				weights_model1 = layer.get_weights()
				weights_model2 = pretrained_layer.get_weights()
				for w1, w2 in zip(weights_model1, weights_model2):
					try:
						tf.debugging.assert_equal(w1, w2)
					except tf.errors.InvalidArgumentError as e:
						self.fail(f"Weights mismatch in embedding layer '{layer.name}': {e}")

	def test_serialization_single_chain_roundtrip(self):
		dataconfig = MultiDataConfigContainer([HUMAN_IGH_OGRDB])
		train_dataset = SingleChainDataset(
			data_path=self.heavy_chain_dataset_path,
			dataconfig=dataconfig,
			use_streaming=True,
			max_sequence_length=576,
		)

		model_params = train_dataset.generate_model_params()
		model = SingleChainAlignAIR(**model_params)
		model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

		dummy_input = {"tokenized_sequence": np.zeros((1, 576), dtype=np.int32)}
		_ = model(dummy_input, training=False)

		with tempfile.TemporaryDirectory() as tmpdir:
			model.save_pretrained(tmpdir)
			reloaded = SingleChainAlignAIR.from_pretrained(tmpdir)
			out_orig = model(dummy_input, training=False)
			out_new = reloaded.predict(dummy_input)
			self.assertIn('v_allele', out_new, 'Reloaded model missing v_allele head')
			self.assertEqual(out_orig['v_allele'].shape, out_new['v_allele'].shape,
							 'Output shape mismatch after reload')
			self.assertTrue(
				np.allclose(out_orig['v_allele'].numpy(), out_new['v_allele'], atol=1e-5),
				'Model outputs diverged after serialization round-trip'
			)

	def test_saved_model_export_single_chain(self):
		dataconfig = MultiDataConfigContainer([HUMAN_IGH_OGRDB])
		ds = SingleChainDataset(
			data_path=self.heavy_chain_dataset_path,
			dataconfig=dataconfig,
			use_streaming=True,
			max_sequence_length=576,
		)
		params = ds.generate_model_params()
		model = SingleChainAlignAIR(**params)
		model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
		dummy = {"tokenized_sequence": np.zeros((1, 576), dtype=np.int32)}
		_ = model(dummy, training=False)
		with tempfile.TemporaryDirectory() as tmpdir:
			model.save_pretrained(tmpdir)
			sm_dir = os.path.join(tmpdir, 'saved_model')
			self.assertTrue(os.path.isdir(sm_dir), 'SavedModel directory missing for single chain')
			loaded = tf.saved_model.load(sm_dir)
			fn = loaded.signatures['serving_default']
			for key in ['v_start','v_end','j_start','j_end','v_allele','j_allele']:
				self.assertIn(key, fn.structured_outputs, f'Missing key {key} in SavedModel outputs')

	@pytest.mark.unit
	def test_single_chain_serialization_config_values(self):
		dataconfig = MultiDataConfigContainer([HUMAN_IGH_OGRDB])
		ds = SingleChainDataset(
			data_path=self.heavy_chain_dataset_path,
			dataconfig=dataconfig,
			use_streaming=True,
			max_sequence_length=576,
		)
		params = ds.generate_model_params()
		model = SingleChainAlignAIR(**params)
		_ = model({"tokenized_sequence": np.zeros((1, 576), dtype=np.int32)}, training=False)
		cfg = model.serialization_config()
		self.assertEqual(cfg['model_type'], 'single_chain')
		self.assertEqual(cfg['max_seq_length'], 576)
		self.assertEqual(cfg['v_allele_count'], model.v_allele_count)
		self.assertEqual(cfg['j_allele_count'], model.j_allele_count)
		self.assertIsNone(cfg['chain_types'])
		self.assertIsNone(cfg['number_of_chains'])

	@pytest.mark.unit
	def test_get_latent_representation_shapes_and_errors(self):
		dataconfig = MultiDataConfigContainer([HUMAN_IGH_OGRDB])
		ds = SingleChainDataset(
			data_path=self.heavy_chain_dataset_path,
			dataconfig=dataconfig,
			use_streaming=True,
			max_sequence_length=576,
		)
		params = ds.generate_model_params()
		model = SingleChainAlignAIR(**params)
		dummy = {"tokenized_sequence": np.zeros((1, 576), dtype=np.int32)}
		_ = model(dummy, training=False)
		v_lat = model.get_latent_representation(dummy, 'V')
		j_lat = model.get_latent_representation(dummy, 'J')
		# Last dim equals Dense units
		self.assertEqual(v_lat.shape[-1], model.v_allele_mid.units)
		self.assertEqual(j_lat.shape[-1], model.j_allele_mid.units)
		# Invalid gene type raises
		with self.assertRaises(ValueError):
			_ = model.get_latent_representation(dummy, 'X')
		# For a chain without D gene, 'D' should raise
		dataconfig_k = MultiDataConfigContainer([HUMAN_IGK_OGRDB])
		ds_k = SingleChainDataset(
			data_path=self.light_chain_dataset_path,
			dataconfig=dataconfig_k,
			use_streaming=True,
			max_sequence_length=576,
		)
		params_k = ds_k.generate_model_params()
		model_k = SingleChainAlignAIR(**params_k)
		_ = model_k(dummy, training=False)
		self.assertFalse(model_k.has_d_gene)
		with self.assertRaises(ValueError):
			_ = model_k.get_latent_representation(dummy, 'D')

	@pytest.mark.unit
	def test_save_pretrained_creates_complete_bundle(self):
		dataconfig = MultiDataConfigContainer([HUMAN_IGH_OGRDB])
		ds = SingleChainDataset(
			data_path=self.heavy_chain_dataset_path,
			dataconfig=dataconfig,
			use_streaming=True,
			max_sequence_length=576,
		)
		params = ds.generate_model_params()
		model = SingleChainAlignAIR(**params)
		# Intentionally don't build first to exercise build-on-save path
		with tempfile.TemporaryDirectory() as tmpdir:
			model.save_pretrained(tmpdir)
			expected = [
				'config.json', 'dataconfig.pkl', 'training_meta.json', 'VERSION', 'README.md', 'fingerprint.txt'
			]
			for fname in expected:
				self.assertTrue(os.path.exists(os.path.join(tmpdir, fname)), f"Missing bundle file: {fname}")
			self.assertTrue(os.path.isdir(os.path.join(tmpdir, 'saved_model')), 'Missing saved_model directory')

	@pytest.mark.unit
	def test_export_saved_model_includes_logits_when_requested(self):
		dataconfig = MultiDataConfigContainer([HUMAN_IGH_OGRDB])
		ds = SingleChainDataset(
			data_path=self.heavy_chain_dataset_path,
			dataconfig=dataconfig,
			use_streaming=True,
			max_sequence_length=576,
		)
		params = ds.generate_model_params()
		model = SingleChainAlignAIR(**params)
		_ = model({"tokenized_sequence": np.zeros((1, 576), dtype=np.int32)}, training=False)
		with tempfile.TemporaryDirectory() as tmpdir:
			model.save_pretrained(tmpdir, include_logits_in_saved_model=True)
			sm_dir = os.path.join(tmpdir, 'saved_model')
			loaded = tf.saved_model.load(sm_dir)
			fn = loaded.signatures['serving_default']
			# Expect at least one logits tensor to be exposed
			logits_keys = [k for k in fn.structured_outputs.keys() if k.endswith('_logits')]
			self.assertTrue(len(logits_keys) > 0, 'Expected logits tensors in SavedModel outputs when requested')

	@pytest.mark.unit
	def test_metrics_contains_compiled_metrics(self):
		dataconfig = MultiDataConfigContainer([HUMAN_IGH_OGRDB])
		ds = SingleChainDataset(
			data_path=self.heavy_chain_dataset_path,
			dataconfig=dataconfig,
			use_streaming=True,
			max_sequence_length=576,
		)
		params = ds.generate_model_params()
		model = SingleChainAlignAIR(**params)
		model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), metrics=[tf.keras.metrics.AUC(name='auc')])
		_ = model({"tokenized_sequence": np.zeros((1, 576), dtype=np.int32)}, training=False)
		metric_names = [m.name for m in model.metrics]
		self.assertIn('auc', metric_names)

	def test_serialization_multi_chain_roundtrip(self):
		multi_configs = MultiDataConfigContainer([HUMAN_IGK_OGRDB, HUMAN_IGL_OGRDB])
		dataset = MultiChainDataset(
			data_paths=[self.light_chain_dataset_path, self.light_chain_dataset_path],
			dataconfigs=multi_configs,
			max_sequence_length=576,
			use_streaming=True,
		)
		params = dataset.generate_model_params()
		model = MultiChainAlignAIR(**params)
		model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
		dummy = {"tokenized_sequence": np.zeros((1, 576), dtype=np.int32)}
		_ = model(dummy, training=False)
		with tempfile.TemporaryDirectory() as tmpdir:
			model.save_pretrained(tmpdir)
			reloaded = MultiChainAlignAIR.from_pretrained(tmpdir)
			out_a = model(dummy, training=False)
			out_b = reloaded.predict(dummy)
			for k in ['v_allele','j_allele','chain_type']:
				self.assertIn(k, out_b, f"Missing output {k} after reload")
				self.assertEqual(out_a[k].shape, out_b[k].shape, f"Shape mismatch for {k}")

	def test_saved_model_export_multi_chain(self):
		multi_configs = MultiDataConfigContainer([HUMAN_IGK_OGRDB, HUMAN_IGL_OGRDB])
		dataset = MultiChainDataset(
			data_paths=[self.light_chain_dataset_path, self.light_chain_dataset_path],
			dataconfigs=multi_configs,
			max_sequence_length=576,
			use_streaming=True,
		)
		params = dataset.generate_model_params()
		model = MultiChainAlignAIR(**params)
		model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
		dummy = {"tokenized_sequence": np.zeros((1, 576), dtype=np.int32)}
		_ = model(dummy, training=False)
		with tempfile.TemporaryDirectory() as tmpdir:
			model.save_pretrained(tmpdir)
			sm_dir = os.path.join(tmpdir, 'saved_model')
			self.assertTrue(os.path.isdir(sm_dir), 'SavedModel directory missing for multi chain')
			loaded = tf.saved_model.load(sm_dir)
			fn = loaded.signatures['serving_default']
			for key in ['v_start','v_end','j_start','j_end','v_allele','j_allele','chain_type']:
				self.assertIn(key, fn.structured_outputs, f'Missing key {key} in multi-chain SavedModel outputs')


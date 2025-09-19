"""Unit tests for utility components in AlignAIR.

This module keeps unittest.TestCase structure for compatibility while improving
clarity, path robustness, and direct-run ergonomics. Tests can be executed via
pytest or by running this file directly.
"""

import os
import sys
import unittest
from typing import List, Tuple

"""
Ensure local package imports work when running this file directly with `python tests/unit/test_utils.py`.
We add <repo_root>/src to sys.path so `import AlignAIR...` resolves to the local source tree.
"""
from pathlib import Path
try:
	REPO_ROOT = Path(__file__).resolve().parents[2]
	SRC = REPO_ROOT / "src"
	if str(SRC) not in sys.path:
		sys.path.insert(0, str(SRC))
except Exception:
	# Best effort; pytest will handle sys.path via tests/conftest.py
	pass

import pytest
import numpy as np
from GenAIRR.dataconfig.enums import ChainType
from AlignAIR.PostProcessing.HeuristicMatching import HeuristicReferenceMatcher

# Load common GenAIRR data configs into globals as in original monolith
from GenAIRR.data import _CONFIG_NAMES
from GenAIRR import data as _gen_data
for _cfg in _CONFIG_NAMES:
	globals()[_cfg] = getattr(_gen_data, _cfg)

from GenAIRR.data import HUMAN_IGH_OGRDB, HUMAN_IGK_OGRDB, HUMAN_IGL_OGRDB, HUMAN_TCRB_IMGT
from AlignAIR.Preprocessing.LongSequence.FastKmerDensityExtractor import FastKmerDensityExtractor
from AlignAIR.Utilities.step_utilities import MultiFileInfoContainer


@pytest.mark.unit
class TestModule(unittest.TestCase):

	def setUp(self):
		"""Common setup; paths are defined for potential future use."""
		self.heavy_chain_dataset_path = 'data/test/sample_igh.csv'
		self.light_chain_dataset_path = 'data/test/sample_igl_k.csv'
		self.tcrb_chain_dataset_path = 'data/test/sample_tcrb.csv'
		self.test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

	def tearDown(self):
		pass

	def test_heuristic_matcher_basic(self):
		"""Heuristic matcher should align simple heavy-chain examples correctly."""
		IGHV_dc = HUMAN_IGH_OGRDB
		IGHV_refence = {i.name:i.ungapped_seq.upper() for j in IGHV_dc.v_alleles for i in IGHV_dc.v_alleles[j]}

		sequences = ['ATCGCAGGTCACCTTGAAGGAGTCTGGTCCTGTGCTGGTGAAACCCACAGAGACCCTCACGCTGACCTGCACCGTCTCTGGGTTCTCACTCAGCAATGCTAGAATGGGTGTGAGCTGGATCCGTCAGCCCCCAGGGAAGGCCCTGGAGTGGCTTGCACACATTTTTTCGAATGACGAAAAATCCTACAGCACATCTCTGAAGAGCAGGCTCACCATCTCCAAGGACACCTCCAAAAGCCAGGTGGTCCTTACCATGACCAATATGGACCCTGTGGACACAGCCACATATTACTGTGCATGGATACATCG',
					 'ATCGCGCAGGTCACCTTGAAGGAGTCTGGTCCTGTGCTGGTGAAACCCACAGAGACCCTCACGCTGACCTGCACCGTCTCTGGGTTCTCACTCAGCAATGCTAGAATGGGTGTGAGCTGGATCCGTCAGCCCCCAGGGAAGGCCCTGGAGTGGCTTGCACACATTTTTTCGAATGACGAAAAATCCTACAGCACATCTCTGAAGAGCAGGCTCACCATCTCCAAGGACACCTCCAAAAGCCAGGTGGTCCTTACCATGACCAATATGGACCCTGTGGACACAGCCACATATTACTGTGCATGGATACATCG',
					 ]
		starts = [3,5]
		ends = [303,305]
		indel_counts = [0,0]

		alleles = ['IGHVF1-G1*01','IGHVF1-G1*01']

		matcher = HeuristicReferenceMatcher(IGHV_refence)
		results = matcher.match(sequences, starts,ends, alleles,indel_counts=indel_counts,_gene='v')

		self.assertEqual(results[0]['start_in_ref'],0)
		self.assertEqual(results[0]['end_in_ref'],301)

		self.assertEqual(results[1]['start_in_ref'], 0)
		self.assertEqual(results[1]['end_in_ref'], 301)

	def test_orientation_classifier(self):
		"""Orientation classifier should predict orientation/transform labels across chains."""
		from AlignAIR.PretrainedComponents import builtin_orientation_classifier
		classifier = builtin_orientation_classifier()
		self.assertIsNotNone(classifier)

		test_sequence_heavy = ['AGAACCCGTCCCTCAAGAGTCGGGTCACCNTATCAATAGACAAGTCCGAGAGCCAGTTCTCTCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTCTGTTCCTGTGCGAGATTCCATATGATAATTGAAAGTCTGCTTTGACCACTGGGACAAGGAAAACCTGGTCNCCGTCTCTTCAGNCTTCTACGACGGAC',
						 'CGCCTNCGGACCCCTCTGCCACTGGGCCCAATGGACCGGGGTCAAGACCTTCACAACGTTATANGTAAGCAACGGTTCATGTGGTTAATTGCCGGCAAGGGAGCCAAAAGTCCGGGGGGTAANAGTGTATGCNGCACAAGAAAGTGAGTACAGACCTCTACCACTCAGACGGAAAGTGGCTGCGTCGTACAGGACAGCGCATTGACAATCAAAAAGAANATTATACCGATTAGGTGAGGTCGGGAAAGGGCCTTCCGACAGCCTGGGTCACGTGTCGTCTCAATTACTTCNACTCGGGTCTCCGACGTGTCCTCTCAAAGTCCCTGGGGGGTCCGGCCTGGTTCAGAGAGGGCCTGAGGTGGTTGCCGTGAAGTGGACGTGANTTTCCGGGAA',
						 'CAAGACCGATAGGCCCTTGGTAGACGTCTGAGGGGACGGTGACCAGGGTGACCTGGNCCCAGTGCTGGAGGNTTGGGGATTCCATCGACGATCTNCACAAATTCGGGTACAATAAGACACGGCCGTGGCTGCGGTTCTCAGGCTGTTAANTTTTAAATACGCCGTGTGCTTTGAATCATTTCTGGAGATGGTGAATATGCCTTTCACCGAGATAGNATAATTCGTCGCACAAGTGTTAGATTTTTTTCTAGTATGGCCAANCCANTCCAGTTCTTTCCCGCAAGTCTGGCGGACCCACGGCAGAGAAGAGCCACTGAGGCTGATCTCAGAGCCTGCACCGCAGAGATTGAGGGAGCCACCAGGCTGGACCAAGTTTTTCCCGGACCCCACCAGGTGCATCTCGGCATGGCGTCTCAAGTTGACGTCNCTTGTGCCCGGTACCCTTTCTNTGGACCAACCGTAGGTGATCTCA',
						 'AAGGTGCAGCTGGTAGAGTCTGGGGGAGGCTTGGTTCAGCCTGGAAGGTCCCTGAGACTCTCCTGTGCAGCCTCTTGATTCACCCTCAATAGCTATGCCATGCACCGGGTCCGCCAGGCTCCAGGCAANGGGCTGGAGTGGCTGGCANTTCTATCATATGATGGAAGTAATAGATCCTATGCAGACACCATGAAGGCCCGACTCACCATTTCCAGAGACAATTGCAANAACANGCTTTATATGCAAGTTAACAGCCTGNGACGTGAGGACACGGCTGTATATTAGTGTGCGAGGGGACTCCTAAGGTCCATAGCATCTGCTAGACTTTGACTACCGGGGCCAGGGAAACCTGGTCACCNTCTCCTCAGACTTCCACCAGGGGCCCATCGGTNTTCCC',
						 'CCCTTGGTGGAGGCCTGAGGAGACGGTGACCAGGGTTCCCTGGCCCCAGTAGTCAAAGTTTTCANCGTAGTCATTTCTCACACAGTAATACACAGCCATGTCCTCGGCTCTCCGTCTGTTCTTTTGCAGATACAGGGAGTTNCTGGAATTGTCTCTGGAGATGATGAATCGGCGCTTCACGGAGTCCACATAGTGAGTCCTACTGCCACTCCAACTAACACCNGATACCCACTCCAGCCCCTTTCCTGGAGCCTTGNAGGCCCAGTTCATGTCANTGTTACTGAAGGTGAATCCAGAGGCTGCACAGGAGAGTCTCAGGGACCCCCNAGGCTGTACCAAGCCTCCCNCAGACTCCACCAGCTGCACCTC',
						 'CCGTCACCACTGTGTTTGATGTGTGTCTTCAAGGTCCGTTCTCAGTGGTAATGGTCGCGACACAGGTGTTCGTGTCGCATGTACCTTGACTCGTCCGATTCTAGGCTCCTGTGCCGGCACATAATGACACGCAGTCCCGAGGNCGATAGGACCCCCTACGCGGAGGGCCAAGCTGGGNACCCCGGTCCNTTGGGNCCAGTGGCAGAGGAGTCCGGAGGTGGTTCCC',
						 'GGGGAAGACCGATGGGCCCTTGGTGGAAGCCTGAGGNGGCAGTGACTAGGGTATTACGGCCTGAGAGCTCGAAGTGCCAGCCTGTAGTACCAGCTGNTGCTATTCGGGCTCAGTAATACAGAGCAGTATCNTTGGATNTGAGGCTGACCATTTGCAGCTACAGTGAGTTCTCCGCCTTGGATCACGATATAGTGACTTGGNACTTCACTGAGTCTGCGTAGTATGACGTACTACCCCTATTACAAATAGGCGAGATCCAATATCGACCCTCACCACACGCCTGGCGGACGCAGTTCACGCTATAGGNACGTAACTTGCAGACAGATCTTGCGCAGGAGAGTCTTAGGCATCCCCGACGCCTGCCCCGGCCTCCCCCAGACTCCAACANCCTGACATCCCCTTATGCCTGCCATCCAGACATTATTTTAATCCAGATTTTTACAACGCTAGCGCCCCANGAGCTACAGTATAACAAGCGGCNTACTTCACGTGCTTAACTNGTAATCGCGCCCGCTAACGTTCACGATGTCGTTCCTAGGT',
						 'GGGAACCACCTCCGGACAAATCNGTCACTGGACCCACGGTGACTCGGTGAGTATCTTTATACTGCCCCCTTTGGGGTGATTCGGGCCAGGCATTGAGAGATTGTCAAACCGCGCCGACCCTGGGGACTGACATNCGATGACTGCGGATGTCTACGGAAACATTGTNTACACAGGAACCTTTTCAGTTTGTCNGGACACTT',
						 'AGTCTCTTCCAGAGGACGTTCCGAAGACCTAAGTGGAAANGATCGAGACGACACGNCAACCACGCTGGCTGAGCACCTGTTGAGGAACTCACCTATCCTACCTAGCAGCAACCGTCACCATTGAGTTNGATGCGTGTCTGCAAGGTCCTTTNTCAGTGGTAATGGTCCCTGTACAGGNGTTCGTGTCGGATGTACCTCGACTCGTCGGACTCTAGGCTCCTNTGCCGGCACATAATGGCACGCCGTCAACCCAGAACTTCACCTATGTCGTCCCCCCTGGTACGAAAACTATAAACCCCTTTTCCCTGTTACCAGTGGCAGAGATTTCCGAAGGTGGTTCCCGGGTAGCCAGAAGGNGGAC',
						 'CCGCGAGGCCAAGCACGAANCGTCCTCCGGTTAGTATACAAGACCTTAGGATATCTCTAGAGTGCCCCCCCTAGTTNGGACTAAACTCCCACTTCGNTTTCCTCAGGTCGAGTCCTGANCGCTTCGTAAGTGNCCGAGACAAGGTGCGGAGATTACAGAATACAGAGAGGTCGACCTCACCAGAAGAGTTACCCTGCTTCTACGCGGTCGGTTGAGCCTTCCCTGAANTCGACCAACGATCCTATCTAGGGTAACTCTCGTGGTTGTTGTGGGCGAGGGAGTTCTCCGCTCTGCGGCACTGTGATCTGTGCACGTGCTTGGTAAAGGAGGCCGTGCACTTNAGACCCCGGCGGCTTCTGTGCCGGCACTTAAAAACACGCTCTTTCCGAACATACATACCNCGCAANCTTTAGACCCCGGGACCGGGATATCACTGTCAGGTATGACCGGAGGTGGTTCCCGGGTAGCCA'
							   ]
		test_labels_heavy = ['Normal',
							 'Reversed',
							 'Reverse Complement',
							 'Normal',
							 'Reverse Complement',
							 'Complement',
							 'Reverse Complement',
							 'Reversed',
							 'Complement',
							 'Complement']

		predicted_heavy = classifier.predict(test_sequence_heavy)
		for predicted, label in zip(predicted_heavy, test_labels_heavy):
			self.assertEqual(predicted, label)

		classifier = builtin_orientation_classifier(ChainType.BCR_LIGHT_LAMBDA)
		light_chain_test = ['CGCCTAGTTTGGTCAAGATATGTGCAATGTNAGCGGTCTGCCTCTGTAGCGTACTTGGTCAAGCTCTACTAGTAGGACAACCCAATTAGCTCCCCCCAGCCGTCTTCTGAGCTGCCTCNGGACCCTGCTGTGTCTGTGGCCTTNGGACAGACAGTCAGGATCACATGCNAAGGAGACAGCCTCAGAAGCTATTATGCAAGGTGGTCCCAGCAGAAGCCAGGGACAGGCCCCTGTACTTGTCATCNGTGGTAAAAACTACCGGCCCTCAAGGATCCCAGACCGATTCTCTGGCTCCAGCTCAGGAAACACAGCTTCCTTGGCCATCACTGGGGCTCAGGCGGATGATGAGGCTGACNATTACTGTANCTCCCGGGACAGCAGTGGTAACCNACTTTGTCTTCGGAACTGGGACCAAGGTCACCGTCCTAGGGTCAGCCCA',
							 'TGGGAACAGAGTGACCGAGGGGGCAGGCTTGCGCTGACCCTCAGGGCGCTGACCTTGGTCCCAGTTCCTANGATTAATGCCCGTACCCCAGGTCTGAGAGTAATAGTCAGCCTTCATCCTCAAACTGGAGGCTGGAGATGGNGATGGCNGTTCTCAGCCCCAGAGCTNGNGTCTGAAGAGCGATCAGGGATCCCGTCCCTCTTGCTGTGACTGCCCTCACTGTTAAGCGTCATCAAGTAGCGAGGCCCCCTCTCTGCTGCTGTTGATGTCATGTGACGGCGTAGCCGGT',
							 'ACNGGCTCCCCCNTCGGAACCCGACTGGGCTCCCGCCAGTGGACCCACGGAGGAGGCTTGTGTGATTGACAGGGTNCAGAGTGTCATTATTAGTCGGAGTAGGAGTCTGACCTCCAACCTCTACCACCCCCTCATCAGTCGGGGTCTCGACCTCGGACTCTTCGCTAGTCCTTGAGGCGAGGGGAACAACATCCAAGGCGATGGAAGTCNTAANTACTTAATGGCTCCCCGGANGGGGCCGACGANACCTACGGTACGCTACTACTTTGATAACACGGGTGACGATTCTGACGTCCACTCTATCTGGCTCCTAGGGTCCCTTTGTCTCCGTCTCCTACTAACTCAGTCCTGTCCGAC',
							 'AGGAAGACAGATGGTGCAGGCACAGTAGGGCTTAATCACCAGTCGGGTCCNTTGGCCGACAGAGGGAAAACATTAAACTGACGACTAGGACTAACTTGCAAAATCGTCAGACTGTAGGCTGGTGANGATAAGAGTGAAGGCTGTCCCAGNTCCACTGTCGC',
							 'GTCGGCCACAACTGAGTNGGTTGAAGGGAGAGTCGNAGAGGACCTCGTAGTCGGTCTGAGTGGACGTGCAAAGNGTCACCGTAATTAGAACCATCGATGTCCTATGAGACCATGGTCGTCTTCGGTGTCTCGGGAGGGGCGAGAGAGGACTCGATNATGAGTCTGAGTTCATTCNTAGTCCCGAGACCTCAGGGGTCGGCGAAGAGACCTAGGTTTCTACGAAGCTCGTTACGTCCCTAAAATCAGTAAAGACCCGAGGTCAGACTTCTACTCCGACTGNTAATGACATACTAAACCGTGTTGTCACGATCCACAAGCCTCCTCCGGGGGTCAACTGGCGGGAGCCCAGNCGGGTTCCGACGGGGGAGCCAGTGA',
							 'GACCGATGGGCCCTTGGTGGAGGCCTGANGANACGGTGAAGGTGGTGCCTACTCTACAGGCNTCTTTCGAGTNGTAGTAATCGTAGGAGGGTAATTCAGAGTCCAGTCCTGGCCCCGATGATCACTCGCACAGTAATACAAGGCCGTGTCCACAGATCTCAGGCTGCTAAGCTGCATGTAGGCTGTGCCNGTGGTATTGTCCATGGTAATCGTGGCTCTGTTCTGGAAGTTCTGTCCGCAGCTTGCTATTCCGAAGGTAGGGAGGATCTCTCCCATGCACTCAAGCGCGTGTTCAGGGTCGTGTCGGACCCACCTGATAGCATAGGTGCTGANGGTGCCTCCAGAATCCGTTNAGNAGACCTTCACTGAGGACCCAGGCTGGTTCATATGAGCCGCAGAGTGCACCAGGAGCACCAC',
							 'GACCAGCCCCNACGGACTCCTCTGCCACTGGTCCCAAGGGACCGGGGTCCCCAGCTTGGTCAACAGGCATATAAGATCGTGAAAGCGTNCCATTATGTGTNGGCACAGGAGTCGAGAGTCCGACAAGTAAACGTCTATGTNGCACAAGAACCTTAACAGAGACNTCTACCACTTAGCCGGGAAGTGCCNCAGACGTATCATAAATAATGAAGGTAGTATACTATATTGACGGTGGGTGAGGTCGGGGAACGGACCTCGGACCGCCTGGGTCACGTACGGTATGGATGANTTCCACTTAGGTCTCCGACGTGACCTCTCAGAGTCCCTGGAGGGTCCGACCTGGTGCGGAGGGGGTCTGAGGTNGTCGACGTGGACGTAGAGGAACTGAGCAGCTAGCGATAATCACACCGCGCCTGG',
							 'CATGNCCGGCAGGTGCAGTCTGAGGCTCAGCGGAAGAAGCCTGGGTCCTCGGGGAANGTCTGCTGCAAGGCTTCTGCCGGAACCTTCTGCAGGTATTCTATCANCCGAGTGCGACAGGCCCCTAGACAAAAGCTNGAGTGGATGGGAGGCATCCTCCTTATCTTCTGTACAGCAAACTACGCACAGATGTTTCATTGNTGACTCACCATTACCACGGACGTATCCNCGAGCACATCCGGCANGGCGCTCAGCAGCCTTAGATGGGATGACACGGCCGTGGATTACTGCGCGAGAGAAGCGGATGTGGCTTTCGGACGCTTTCTATTATTGGGGCAATGGGACCATGTTCTCCGTCTCTTCAGGCCTCCA',
							 'AACAGGAAGACTGATGGTCCCCCCAGGAACTCAGGTGCCTGAGGAGACGGTGACCGTGGTCCCTTTGTCTCATACANGCAAGTCGTCGGCATCCCTTCCAGCTACCACTACTAGAATCCCCCTTTAAGGCTCTCGCAAGTAATACACGGCCGTGTCCGCAGATGTCACAGAGCTGCAGCATCAGGGTTAGCTGGATCTTGGACGTGTATACTGATANGGTGACTCGACTCTTGCGGGAGGGGTTGTAGTTGGCGNTTCCACTGCTATAGCTATACCCNATGCACCCCAGTCCCTTCCNTGGGCGCTGCCGGAGCCGGCTTAAGTAGTGACGACTTAAGGAGCGACCGGNGACAGTGCAGGTGAGGACATGGTCTCCGAAGGGTTCACCAGTCCCGCGCCCGANTCCTGCAGCTGNACCTGGCCGCTCCCCTTATGCGGACTGTTGAGCACGATAGAGNCAGACTAACCTCTC',
							 'CCCTTCTGGCTANCCGGGAACCACCTTCGGACTCCGCTGCCACTGGCACGAGGGACACGGGGTCTGCAGGGACATCATCTGAACCTCAAGGGTGGCATCAGGGTCTGAAAGGTCATGTCAATTTTCGACGGTACNAGACCAAAGACTCCTAATACGAAACGGACATGTCGTACATGTACCACAACAAAGACTTACACCATTCAGACGGGNATCGTATCATCCGTGAAATAACTACTGAAAGTGGTGCGGTATATTNCCGTAGGGTAAATTCGGCGAAGGGACTCCGGACCGTTTGGATCANTTGGAGTATCTATGACTTCCTCTTGGCTCTCCGACGTGTCCTCTCGGAGTCCCTCGGNAGATTCACATGGTTCAGAGGGGGTCTGAGCCATTCAACCCGGGGTAAGGAAAGGCAGGAAANAGACGCCCAATACGTACCGCATAAGTGNTCCGAGTANGCTTTTCCGCAATCCACT'
							]
		lightchain_labels =['Normal',
							 'Reverse Complement',
							 'Reversed',
							 'Reverse Complement',
							 'Complement',
							 'Reverse Complement',
							 'Reversed',
							 'Normal',
							 'Reverse Complement',
							 'Reversed'
							]
		predicted_lightchain = classifier.predict(light_chain_test)
		for predicted, label in zip(predicted_lightchain, lightchain_labels):
			self.assertEqual(predicted, label)

	def test_file_name_extraction_step(self):
		"""FileNameExtractionStep should populate FileInfo from input path."""
		from AlignAIR.Preprocessing.Steps.file_steps import FileNameExtractionStep
		from AlignAIR.Utilities.step_utilities import FileInfo
		from unittest.mock import Mock

		mock_predict_object = Mock()
		mock_predict_object.script_arguments.sequences = 'path/to/sequences/file.csv'

		step = FileNameExtractionStep("Extract File Name")
		result = step.process(mock_predict_object)

		self.assertIsInstance(result.file_info, FileInfo)
		self.assertEqual(result.file_info.file_name, 'file')
		self.assertEqual(result.file_info.file_type, 'csv')
		self.assertEqual(result, mock_predict_object)

	def test_file_sample_counter_step(self):
		"""FileSampleCounterStep should use the appropriate row counter and store count."""
		from AlignAIR.Preprocessing.Steps.file_steps import FileSampleCounterStep
		from AlignAIR.Utilities.file_processing import FILE_ROW_COUNTERS
		from AlignAIR.Utilities.step_utilities import FileInfo
		from unittest.mock import Mock

		mock_predict_object = Mock()
		file_info = FileInfo('./tests/sample_HeavyChain_dataset.csv')
		mock_predict_object.file_info = file_info
		mock_predict_object.script_arguments.sequences = './tests/sample_HeavyChain_dataset.csv'

		mock_row_counter = Mock(return_value=100)
		FILE_ROW_COUNTERS['csv'] = mock_row_counter

		step = FileSampleCounterStep("Count Samples")
		result = step.process(mock_predict_object)

		mock_row_counter.assert_called_with('./tests/sample_HeavyChain_dataset.csv')
		self.assertEqual(file_info.sample_count, 100)
		self.assertEqual(result, mock_predict_object)

	def test_fast_kmer_density_extractor(self):
		"""FastKmerDensityExtractor should recover true sequence within mismatch tolerance."""
		ref_alleles = (list(map(lambda x: x.ungapped_seq.upper(), HUMAN_IGH_OGRDB.allele_list('v')))+
						  list(map(lambda x: x.ungapped_seq.upper(), HUMAN_IGH_OGRDB.allele_list('d')))+
							list(map(lambda x: x.ungapped_seq.upper(), HUMAN_IGH_OGRDB.allele_list('j')))
					   )

		candidate_sequence_extractor = FastKmerDensityExtractor(11, max_length=576, allowed_mismatches=0)
		candidate_sequence_extractor.fit(ref_alleles)

		import json
		json_path = Path(REPO_ROOT, 'tests', 'data', 'misc', 'KMer_Density_Extractor_HeavyChainTests.json')
		with open(json_path, 'r') as f:
			tests = json.load(f)

		results = [candidate_sequence_extractor.transform_holt(noise)[0] for true, noise in tests]

		def is_within_mismatch_limit(true_seq: str, result_seq: str, max_mismatch: int = 5) -> bool:
			true_len = len(true_seq)
			for i in range(max_mismatch + 1):
				if true_seq[i:] in result_seq:
					return True
				if true_seq[:true_len - i] in result_seq:
					return True
			return False

		for s, (result, (true, _)) in enumerate(zip(results, tests)):
			self.assertTrue(len(result) <= 576, f"Test {s} failed: Result length exceeds 576")
			self.assertTrue(is_within_mismatch_limit(true, result),
							f"Test {s} failed: True sequence not found within mismatch limit")

	def test_heuristic_matcher_comprehensive(self):
		"""Heuristic matcher should handle diverse overhang/mutation scenarios."""
		ref_seq  = "ATGCGTACGTCAGTACGTCAGTACGTTAGC"
		allele   = "TEST*01"
		matcher  = HeuristicReferenceMatcher({allele: ref_seq})

		def mutate(seq: str, pos: int, base: str) -> str:
			return seq[:pos] + base + seq[pos+1:]

		cases = [
			("exact",               lambda: ref_seq,                        0,     0,        30),
			("overhang_left",       lambda: "GGG" + ref_seq,                0,     0,        30),
			("overhang_right",      lambda: ref_seq + "TTTT",               0,     0,        30),
			("overhang_both",       lambda: "AA"  + ref_seq + "CC",         0,     0,        30),
			("del_left",            lambda: ref_seq[5:],                    5,     5,        30),
			("del_right",           lambda: ref_seq[:-6],                   6,     0,        24),
			("del_both",            lambda: ref_seq[3:-4],                  7,     3,        26),
			("internal_insert",     lambda: ref_seq[:10] + "NNNN" + ref_seq[10:], 4, 0,     30),
			("two_mismatches",      lambda: mutate(mutate(ref_seq, 5,"A"), 15,"C"), 0, 0,   30),
			("mixed_combo",
			 lambda: "GG" + ref_seq[:12] + "NN" + mutate(ref_seq[12:], 5,"G"), 2, 0,        30),
		]

		sequences, starts, ends, alleles, indel_counts = [], [], [], [], []
		for _, builder, indels, *_ in cases:
			seg = builder()
			sequences.append(seg)
			starts.append(0)
			ends.append(len(seg))
			alleles.append(allele)
			indel_counts.append(indels)

		results = matcher.match(
			sequences, starts, ends, alleles,
			indel_counts, _gene="v"
		)

		for (name, _, _, exp_start, exp_end), res in zip(cases, results):
			with self.subTest(case=name):
				self.assertEqual(
					res["start_in_ref"], exp_start,
					f"{name}: start_in_ref mismatch (got {res['start_in_ref']}, want {exp_start})"
				)
				self.assertEqual(
					res["end_in_ref"],   exp_end,
					f"{name}: end_in_ref mismatch (got {res['end_in_ref']}, want {exp_end})"
				)

	def test_heuristic_matcher_numpy_indel_and_overhang(self):
		"""Support numpy scalar indel counts and ensure overhang logic respects nonzero indels."""
		ref = "A" * 20 + "CCCC" + "G" * 20
		allele = "REF*01"
		matcher = HeuristicReferenceMatcher({allele: ref})
		# Build a segment with a 2-base insertion relative to ref around the 'CCCC' area
		seg = ref[:22] + "TT" + ref[22:]
		start, end = 0, len(seg)
		# Use numpy scalar for indels to hit int conversion path
		indels_np = np.array(2)
		res = matcher.match([seg], [start], [end], [allele], [indels_np], k=8, s=10, _gene="v")[0]
		# With indels > 0 we should not force trim to full-length ref; mapping should be inside ref bounds
		self.assertGreaterEqual(res["start_in_ref"], 0)
		self.assertLessEqual(res["end_in_ref"], len(ref))
		self.assertEqual(res["end_in_ref"] - res["start_in_ref"], len(ref))

	def test_calculate_pad_size(self):
		"""Static pad size computes half of remaining length (floor)."""
		seq = "ACGT" * 10  # length 40
		pad = HeuristicReferenceMatcher.calculate_pad_size(seq, max_length=100)
		self.assertEqual(pad, (100 - len(seq)) // 2)

	def test_fast_tail_head_check_thresholds(self):
		"""Head/tail k-mer check adheres to mismatch threshold behavior."""
		ref = "A" * 10 + "C" * 10
		# Exactly matches: should be True
		self.assertTrue(HeuristicReferenceMatcher._fast_tail_head_check(ref, ref, k=5, max_mm=0))
		# Within threshold: 1 mismatch at head and tail combined within max_mm
		seg_ok = "T" + ref[1:-1] + "G"  # one mismatch at head, one at tail
		self.assertTrue(HeuristicReferenceMatcher._fast_tail_head_check(seg_ok, ref, k=5, max_mm=2))
		# Exceed threshold: too many mismatches
		seg_bad = "T" * 5 + ref[5:-5] + "G" * 5
		self.assertFalse(HeuristicReferenceMatcher._fast_tail_head_check(seg_bad, ref, k=5, max_mm=2))

	def test_clip_overhang_no_indel_even_odd(self):
		"""Overhang trimming splits excess across both sides (even vs odd)."""
		matcher = HeuristicReferenceMatcher({"R": "A" * 10})
		# Even excess: 14 seg vs 10 ref → excess=4, trim 2 from each side
		start, end = matcher._clip_overhang_no_indel("N" * 20, start=0, end=14, ref_len=10)
		self.assertEqual((start, end), (2, 12))
		# Odd excess: 13 seg vs 10 ref → excess=3, trim 1 from start and 2 from end
		start2, end2 = matcher._clip_overhang_no_indel("N" * 20, start=0, end=13, ref_len=10)
		self.assertEqual((start2, end2), (1, 11))

	def test_affine_alignment_cost_ordering(self):
		"""Affine cost should prefer sequences with more matches (lower score)."""
		matcher = HeuristicReferenceMatcher({"R": ""})
		ref = "ACGTACGT"
		best = matcher._affine_alignment_cost(ref, ref)
		one_mm = matcher._affine_alignment_cost(ref, ref[:-1] + "A")
		self.assertLess(best, one_mm)

	def test_is_pure_overhang_cases(self):
		"""is_pure_overhang follows documented conditions."""
		self.assertTrue(HeuristicReferenceMatcher._is_pure_overhang(12, 10, 0))
		self.assertTrue(HeuristicReferenceMatcher._is_pure_overhang(10, 10, 0))
		self.assertFalse(HeuristicReferenceMatcher._is_pure_overhang(10, 10, 1))
		self.assertFalse(HeuristicReferenceMatcher._is_pure_overhang(8, 10, 0))

	def test_match_quick_exit_same_length_no_indels(self):
		"""When seg==ref and indels==0 and heads/tails match, quick-exit path applies."""
		ref = "ACGTACGTAC"
		matcher = HeuristicReferenceMatcher({"A*01": ref})
		res = matcher.match([ref], [0], [len(ref)], ["A*01"], [0], _gene="v")[0]
		self.assertEqual(res["start_in_ref"], 0)
		self.assertEqual(res["end_in_ref"], len(ref))
		self.assertEqual(res["start_in_seq"], 0)
		self.assertEqual(res["end_in_seq"], len(ref))

	def test_match_overhang_seq_bounds_adjustment(self):
		"""Overhang with zero indels should trim seq bounds to match ref length."""
		ref = "G" * 10
		seg = "TT" + ref + "AA"  # overhang of 4 (2 left, 2 right)
		matcher = HeuristicReferenceMatcher({"R": ref})
		res = matcher.match([seg], [0], [len(seg)], ["R"], [0], _gene="v")[0]
		# Trimmed equally: start_in_seq=2, end_in_seq=len(seg)-2
		self.assertEqual(res["start_in_seq"], 2)
		self.assertEqual(res["end_in_seq"], len(seg) - 2)
		self.assertEqual(res["start_in_ref"], 0)
		self.assertEqual(res["end_in_ref"], len(ref))

	def test_match_unknown_allele_raises(self):
		"""Unknown allele key should raise KeyError from reference lookup."""
		matcher = HeuristicReferenceMatcher({})
		with self.assertRaises(KeyError):
			_ = matcher.match(["AC"], [0], [2], ["UNKNOWN*00"], [0], _gene="v")

	def test_fast_kmer_extractor_mismatch_variants(self):
		"""When allowed_mismatches>0, k-mer variants should be present in the set."""
		ref = ["ACGTAC"]
		extractor = FastKmerDensityExtractor(k=3, max_length=6, allowed_mismatches=1)
		extractor.fit(ref)
		# Variant with one mismatch of "ACG" such as "ACA" should exist
		self.assertIn("ACA", extractor.kmer_set)
		# Transform returns a region of requested max_length (or shorter if sequence shorter)
		region, hist = extractor.transform_holt("TTTACGTAC")
		self.assertTrue(1 <= len(hist))
		self.assertEqual(len(region), extractor.max_length if len("TTTACGTAC") >= extractor.max_length else len("TTTACGTAC"))

	def test_fast_kmer_extractor_no_variants_when_zero_mismatch(self):
		"""When allowed_mismatches=0, variants should not be present in the set."""
		extractor = FastKmerDensityExtractor(k=3, max_length=6, allowed_mismatches=0)
		extractor.fit(["ACGTAC"])
		self.assertIn("ACG", extractor.kmer_set)
		self.assertNotIn("ACA", extractor.kmer_set)

	def test_filename_extraction_multiple_files(self):
		"""FileNameExtractionStep should create a MultiFileInfoContainer for comma-separated inputs."""
		from AlignAIR.Preprocessing.Steps.file_steps import FileNameExtractionStep
		from unittest.mock import Mock
		mock_predict_object = Mock()
		mock_predict_object.script_arguments.sequences = 'a.csv, b.tsv'
		step = FileNameExtractionStep("Extract Names")
		res = step.process(mock_predict_object)
		self.assertIsInstance(res.file_info, MultiFileInfoContainer)
		self.assertEqual(res.file_info.file_names(), ["a", "b"])
		self.assertEqual(res.file_info.file_types(), ["csv", "tsv"])

	def test_file_sample_counter_multiple_files(self):
		"""FileSampleCounterStep should count each file and store per-file sample_count."""
		from AlignAIR.Preprocessing.Steps.file_steps import FileSampleCounterStep
		from AlignAIR.Utilities.file_processing import FILE_ROW_COUNTERS
		from unittest.mock import Mock
		# Prepare predict object with MultiFileInfoContainer
		mock_predict_object = Mock()
		mock_predict_object.file_info = MultiFileInfoContainer('a.csv, b.tsv')
		# Also mirror single-field for backward compat path (not used in multi)
		mock_predict_object.script_arguments.sequences = 'a.csv, b.tsv'
		# Mock counters for extensions
		FILE_ROW_COUNTERS['csv'] = Mock(return_value=10)
		FILE_ROW_COUNTERS['tsv'] = Mock(return_value=5)
		step = FileSampleCounterStep("Count Samples")
		res = step.process(mock_predict_object)
		# Validate counts
		self.assertEqual(res.file_info[0].sample_count, 10)
		self.assertEqual(res.file_info[1].sample_count, 5)
		self.assertEqual(res.file_info.total_sample_count(), 15)

	def test_file_sample_counter_unknown_extension_raises(self):
		"""If file type has no registered counter, a KeyError is raised."""
		from AlignAIR.Preprocessing.Steps.file_steps import FileSampleCounterStep
		from unittest.mock import Mock
		mock_predict_object = Mock()
		mock_predict_object.file_info = MultiFileInfoContainer('x.unknown')
		step = FileSampleCounterStep("Count")
		with self.assertRaises(KeyError):
			_ = step.process(mock_predict_object)

	def test_fileinfo_invalid_inputs(self):
		"""FileInfo should validate path input type and emptiness."""
		from AlignAIR.Utilities.step_utilities import FileInfo
		with self.assertRaises(ValueError):
			_ = FileInfo("")
		with self.assertRaises(ValueError):
			_ = FileInfo(None)  # type: ignore[arg-type]

	def test_multifileinfo_single_proxy_and_len(self):
		"""Single-path container proxies attributes and reports length 1."""
		c = MultiFileInfoContainer('alpha.csv')
		self.assertEqual(len(c), 1)
		# Proxy attributes should resolve from the sole FileInfo
		self.assertEqual(c.file_name, 'alpha')
		self.assertEqual(c.file_type, 'csv')

	def test_multifileinfo_attr_error_on_multi(self):
		"""Accessing proxied attributes on multi-file container should raise AttributeError."""
		c = MultiFileInfoContainer('a.csv, b.csv')
		with self.assertRaises(AttributeError):
			_ = c.file_name

	def test_multifileinfo_invalid_inputs(self):
		"""Invalid MultiFileInfoContainer inputs should raise ValueError."""
		with self.assertRaises(ValueError):
			_ = MultiFileInfoContainer('')
		with self.assertRaises(ValueError):
			_ = MultiFileInfoContainer(' , , ')


if __name__ == "__main__":
	unittest.main(verbosity=2)


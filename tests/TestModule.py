import os

from src.AlignAIR.Metadata import RandomDataConfigGenerator
from src.AlignAIR.Models.LightChain import LightChainAlignAIRR

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import unittest
import pandas as pd
from importlib import resources
from GenAIRR.data import builtin_heavy_chain_data_config,builtin_kappa_chain_data_config,builtin_lambda_chain_data_config
from src.AlignAIR.Data import HeavyChainDataset, LightChainDataset
from src.AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
from src.AlignAIR.Trainers import Trainer
import tensorflow as tf
import numpy as np
from src.AlignAIR.PostProcessing.HeuristicMatching import HeuristicReferenceMatcher

# Define a test case class inheriting from unittest.TestCase
class TestModule(unittest.TestCase):

    # Define a setup method if you need to prepare anything before each test method (optional)
    def setUp(self):
        self.heavy_chain_dataset_path = './sample_HeavyChain_dataset.csv'
        self.light_chain_dataset_path = './sample_LightChain_dataset.csv'
        self.heavy_chain_dataset = pd.read_csv(self.heavy_chain_dataset_path)
        self.light_chain_dataset = pd.read_csv(self.light_chain_dataset_path)


    # Define a teardown method if you need to clean up after each test method (optional)
    def tearDown(self):
        # Teardown code here, e.g., closing files or connections
        pass

    # Define your test methods, each starting with 'test_'
    def test_heavy_chain_model_training(self):
        train_dataset = HeavyChainDataset(data_path=self.heavy_chain_dataset_path,
                                          dataconfig=builtin_heavy_chain_data_config(),batch_read_file=True,
                                          max_sequence_length=576)

        trainer = Trainer(
            model=HeavyChainAlignAIRR,
            dataset=train_dataset,
            epochs=1,
            steps_per_epoch=max(1, train_dataset.data_length // 10),
            verbose=1,
            classification_metric=[tf.keras.metrics.AUC(), tf.keras.metrics.AUC(), tf.keras.metrics.AUC()],
            regression_metric=tf.keras.losses.binary_crossentropy,
            optimizers_params={"clipnorm": 1},
        )

        # Train the model
        trainer.train()

        self.assertIsNotNone(trainer.history)

    def test_light_chain_model_training(self):
        train_dataset = LightChainDataset(data_path=self.light_chain_dataset_path,
                                          lambda_dataconfig=builtin_lambda_chain_data_config(),
                                          kappa_dataconfig=builtin_kappa_chain_data_config(),
                                          batch_read_file=True,
                                          max_sequence_length=576)

        trainer = Trainer(
            model=LightChainAlignAIRR,
            dataset=train_dataset,
            epochs=1,
            steps_per_epoch=max(1, train_dataset.data_length // 10),
            verbose=1,
            classification_metric=[tf.keras.metrics.AUC(), tf.keras.metrics.AUC(), tf.keras.metrics.AUC()],
            regression_metric=tf.keras.losses.binary_crossentropy,
            optimizers_params={"clipnorm": 1},
        )

        # Train the model
        trainer.train()

        self.assertIsNotNone(trainer.history)

    def test_load_saved_heavy_chain_model(self):
        train_dataset = HeavyChainDataset(data_path=self.heavy_chain_dataset_path
                                                      , dataconfig=builtin_heavy_chain_data_config(),
                                                      batch_size=32,
                                                      max_sequence_length=576,
                                                      batch_read_file=True)

        trainer = Trainer(
            model=HeavyChainAlignAIRR,
            dataset=train_dataset,
            epochs=1,
            steps_per_epoch=1,
            verbose=1,
        )
        trainer.model.build({'tokenized_sequence': (576, 1)})

        MODEL_CHECKPOINT = './AlignAIRR_S5F_OGRDB_Experimental_New_Loss_V7'
        trainer.model.load_weights(MODEL_CHECKPOINT)
        self.assertNotEqual(trainer.model.log_var_v_end.weights[0].numpy(),0.0)

        seq = 'CAGCCACAACTGAACTGGTCAAGTCCAGGACTGGTGAATACCTCGCAGACCGTCACACTCACCCTTGCCGTGTCCGGGGACCGTGTCTCCAGAACCACTGCTGTTTGGAAGTGGAGGGGTCAGACCCCATCGCGAGGCCTTGCGTGGCTGGGAAGGACCTACNACAGTTCCAGGTGATTTGCTAACAACGAAGTGTCTGTGAATTGTTNAATATCCATGAACCCAGACGCATCCANGGAACGGNTCTTCCTGCACCTGAGGTCTGGGGCCTTCGACGACACGGCTGTACATNCGTGAGAAAGCGGTGACCTCTACTAGGATAGTGCTGAGTACGACTGGCATTACGCTCTCNGGGACCGTGCCACCCTTNTCACTGCCTCCTCGG'
        es = trainer.train_dataset.encode_and_equal_pad_sequence(seq)[0]
        predicted = trainer.model.predict({'tokenized_sequence':np.vstack([es])})
        #print(predicted)

        self.assertIsNotNone(predicted)


    def test_heuristic_matcher_basic(self):
            # Test case where there is an indel, causing the segment and reference lengths to differ
            IGHV_dc = builtin_heavy_chain_data_config()
            IGHV_refence = {i.name:i.ungapped_seq.upper() for j in IGHV_dc.v_alleles for i in IGHV_dc.v_alleles[j]}

            sequences = ['ATCGCAGGTCACCTTGAAGGAGTCTGGTCCTGTGCTGGTGAAACCCACAGAGACCCTCACGCTGACCTGCACCGTCTCTGGGTTCTCACTCAGCAATGCTAGAATGGGTGTGAGCTGGATCCGTCAGCCCCCAGGGAAGGCCCTGGAGTGGCTTGCACACATTTTTTCGAATGACGAAAAATCCTACAGCACATCTCTGAAGAGCAGGCTCACCATCTCCAAGGACACCTCCAAAAGCCAGGTGGTCCTTACCATGACCAATATGGACCCTGTGGACACAGCCACATATTACTGTGCATGGATACATCG',
                         'ATCGCGCAGGTCACCTTGAAGGAGTCTGGTCCTGTGCTGGTGAAACCCACAGAGACCCTCACGCTGACCTGCACCGTCTCTGGGTTCTCACTCAGCAATGCTAGAATGGGTGTGAGCTGGATCCGTCAGCCCCCAGGGAAGGCCCTGGAGTGGCTTGCACACATTTTTTCGAATGACGAAAAATCCTACAGCACATCTCTGAAGAGCAGGCTCACCATCTCCAAGGACACCTCCAAAAGCCAGGTGGTCCTTACCATGACCAATATGGACCCTGTGGACACAGCCACATATTACTGTGCATGGATACATCG',
                         ]
            starts = [3,5]
            ends = [303,305]


            alleles = ['IGHVF1-G1*01','IGHVF1-G1*01']

            matcher = HeuristicReferenceMatcher(IGHV_refence)
            results = matcher.match(sequences, starts,ends, alleles)

            self.assertEqual(results[0]['start_in_ref'],0)
            self.assertEqual(results[0]['end_in_ref'],301)

            self.assertEqual(results[1]['start_in_ref'], 0)
            self.assertEqual(results[1]['end_in_ref'], 301)


if __name__ == '__main__':
    unittest.main()

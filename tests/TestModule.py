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
import os
import subprocess
import shutil
# Define a test case class inheriting from unittest.TestCase
class TestModule(unittest.TestCase):

    # Define a setup method if you need to prepare anything before each test method (optional)
    def setUp(self):
        self.heavy_chain_dataset_path = './sample_HeavyChain_dataset.csv'
        self.light_chain_dataset_path = './sample_LightChain_dataset.csv'
        self.heavy_chain_dataset = pd.read_csv(self.heavy_chain_dataset_path)
        self.light_chain_dataset = pd.read_csv(self.light_chain_dataset_path)
        self.test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))


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
            results = matcher.match(sequences, starts,ends, alleles,_gene='v')

            self.assertEqual(results[0]['start_in_ref'],0)
            self.assertEqual(results[0]['end_in_ref'],301)

            self.assertEqual(results[1]['start_in_ref'], 0)
            self.assertEqual(results[1]['end_in_ref'], 301)

    def test_predict_script(self):

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'AlignAIR', 'API'))
        script_path = os.path.join(base_dir, 'AlignAIRRPredict.py')


        # Ensure the script exists
        self.assertTrue(os.path.exists(script_path), "Predict script not found at path: " + script_path)

        # Define the command with absolute paths
        command = [
            'C:/Users/tomas/Desktop/AlignAIRR/AlignAIR_ENV/Scripts/python', script_path,
            '--model_checkpoint', os.path.join(self.test_dir, 'AlignAIRR_S5F_OGRDB_Experimental_New_Loss_V7'),
            '--save_path', str(self.test_dir)+'/',
            '--chain_type', 'heavy',
            '--sequences', self.heavy_chain_dataset_path,
            '--batch_size', '32'
        ]

        # Execute the script
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, "Script failed to run with error: " + result.stderr)


        # Check if the CSV was created
        file_name = self.heavy_chain_dataset_path.split('/')[-1].split('.')[0]
        save_name = file_name + '_alignairr_results.csv'

        output_csv = os.path.join(self.test_dir, save_name)
        self.assertTrue(os.path.isfile(output_csv), "Output CSV file not created")

        # Read the output CSV and validate its contents
        df = pd.read_csv(output_csv)
        self.assertFalse(df.empty, "Output CSV is empty")
        # Additional assertions can be added here

        # Cleanup
        os.remove(output_csv)

    def test_train_model_script(self):
        # Set base directory and script path
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'AlignAIR', 'API'))
        script_path = os.path.join(base_dir, 'TrainModel.py')  # Name of your training script

        # Ensure the script exists
        self.assertTrue(os.path.exists(script_path), "Training script not found at path: " + script_path)

        # Define the command with absolute paths
        command = [
            'C:/Users/tomas/Desktop/AlignAIRR/AlignAIR_ENV/Scripts/python', script_path,
            '--chain_type', 'heavy',
            '--train_dataset', self.heavy_chain_dataset_path,
            '--session_path', './',
            '--epochs', '1',
            '--batch_size', '32',
            '--steps_per_epoch', '32',
            '--max_sequence_length', '576',
            '--model_name', 'TestModel'
        ]

        # Execute the script
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, "Training script failed to run with error: " + result.stderr)

        # Check if the model weights were created
        models_path = os.path.join('./', 'saved_models', 'TestModel')
        expected_weights_path = os.path.join(models_path, 'TestModel_weights_final_epoch.data-00000-of-00001')
        self.assertTrue(os.path.isfile(expected_weights_path), "Model weights file not created")

        # Optionally, check for the log file
        expected_log_path = os.path.join('./', 'TestModel.csv')
        self.assertTrue(os.path.isfile(expected_log_path), "Log file not created")

        # Cleanup
        if os.path.exists(models_path):
            shutil.rmtree(os.path.join('./', 'saved_models'))  # Remove the saved_models directory and all its contents
        if os.path.exists(expected_log_path):
            os.remove(expected_log_path)  # Remove the log file if it exists


if __name__ == '__main__':
    unittest.main()

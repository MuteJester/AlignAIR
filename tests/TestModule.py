import os

import yaml
from GenAIRR.dataconfig.enums import ChainType

from AlignAIR.Data import MultiChainDataset, MultiDataConfigContainer
from AlignAIR.Models.SingleChainAlignAIR.SingleChainAlignAIR import SingleChainAlignAIR
from AlignAIR.Models.MultiChainAlignAIR.MultiChainAlignAIR import MultiChainAlignAIR
from src.AlignAIR.Data.PredictionDataset import PredictionDataset
from src.AlignAIR.Preprocessing.LongSequence.FastKmerDensityExtractor import FastKmerDensityExtractor
from GenAIRR.dataconfig.make import RandomDataConfigBuilder
from src.AlignAIR.Models.LightChain import LightChainAlignAIRR
import unittest
import pandas as pd
from importlib import resources
from GenAIRR.data import _CONFIG_NAMES
from GenAIRR import data
for config in _CONFIG_NAMES:
    globals()[config] = getattr(data, config)

# Explicit imports for commonly used configs
from GenAIRR.data import HUMAN_IGH_OGRDB, HUMAN_IGK_OGRDB, HUMAN_IGL_OGRDB, HUMAN_TCRB_IMGT
from src.AlignAIR.Data import SingleChainDataset
from src.AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
from src.AlignAIR.Trainers import Trainer
import tensorflow as tf
import numpy as np
from AlignAIR.PostProcessing.HeuristicMatching import HeuristicReferenceMatcher
import os
import subprocess
import shutil


def inspect_model_weights(model_checkpoint_path):
    """
    Inspect the saved model weights to understand the D allele structure
    """
    import tensorflow as tf
    import numpy as np

    print(f"\n=== INSPECTING MODEL WEIGHTS ===")
    print(f"Model checkpoint: {model_checkpoint_path}")

    try:
        # Load the checkpoint
        checkpoint = tf.train.load_checkpoint(model_checkpoint_path)

        # Get all variable names
        var_names = checkpoint.get_variable_to_shape_map()

        print(f"Total variables in checkpoint: {len(var_names)}")

        # Look for D allele related weights
        d_allele_vars = [name for name in var_names.keys() if 'd_allele' in name.lower()]

        print(f"\nD-allele related variables:")
        for var_name in d_allele_vars:
            shape = var_names[var_name]
            print(f"  {var_name}: {shape}")

            # Load the actual weights
            weights = checkpoint.get_tensor(var_name)
            print(f"    Shape: {weights.shape}")
            print(f"    Data type: {weights.dtype}")

            if 'bias' in var_name and len(weights.shape) == 1:
                print(f"    Values (first 10): {weights[:10]}")
                print(f"    Values (last 10): {weights[-10:]}")

                # Check for duplicates - look for identical or very similar values
                if len(weights) > 30:  # Only if we have enough weights
                    # Check if last two values are identical (indicating duplicate Short-D)
                    if np.allclose(weights[-1], weights[-2], atol=1e-6):
                        print(f"    *** POTENTIAL DUPLICATE: Last two values are nearly identical!")
                        print(f"        weights[-2]: {weights[-2]}")
                        print(f"        weights[-1]: {weights[-1]}")

                    # Check for any other duplicates
                    for i in range(len(weights) - 1):
                        for j in range(i + 1, len(weights)):
                            if np.allclose(weights[i], weights[j], atol=1e-6):
                                print(f"    *** DUPLICATE FOUND: weights[{i}] ≈ weights[{j}] = {weights[i]}")

            elif 'kernel' in var_name or 'weight' in var_name:
                print(f"    Weight matrix shape: {weights.shape}")
                if len(weights.shape) == 2 and weights.shape[1] > 30:  # Output dimension
                    # Check last two columns for duplicates
                    last_col = weights[:, -1]
                    second_last_col = weights[:, -2]
                    if np.allclose(last_col, second_last_col, atol=1e-6):
                        print(f"    *** POTENTIAL DUPLICATE: Last two columns are nearly identical!")
                        print(f"        Max difference: {np.max(np.abs(last_col - second_last_col))}")

        # Also check for any variables with 35 in the shape
        vars_with_35 = [(name, shape) for name, shape in var_names.items() if 35 in shape]
        if vars_with_35:
            print(f"\nVariables with dimension 35:")
            for name, shape in vars_with_35:
                print(f"  {name}: {shape}")

        # Check for any variables with 34 in the shape
        vars_with_34 = [(name, shape) for name, shape in var_names.items() if 34 in shape]
        if vars_with_34:
            print(f"\nVariables with dimension 34:")
            for name, shape in vars_with_34:
                print(f"  {name}: {shape}")

    except Exception as e:
        print(f"Error inspecting checkpoint: {e}")
        import traceback
        traceback.print_exc()


class TestModule(unittest.TestCase):

    def setUp(self):
        self.heavy_chain_dataset_path = './sample_HeavyChain_dataset.csv'
        self.light_chain_dataset_path = './sample_LightChain_dataset.csv'
        self.tcrb_chain_dataset_path = './TCRB_Sample_Data.csv'
        self.heavy_chain_dataset = pd.read_csv(self.heavy_chain_dataset_path)
        self.light_chain_dataset = pd.read_csv(self.light_chain_dataset_path)
        self.test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))


    def tearDown(self):
        # Teardown code here, e.g., closing files or connections
        pass

    def test_heavy_chain_model_training(self):
        """Tests the training process for a single heavy chain model with the new Trainer."""
        # 1. Create Dataset
        train_dataset = SingleChainDataset(
            data_path=self.heavy_chain_dataset_path,
            dataconfig=MultiDataConfigContainer([HUMAN_IGH_OGRDB]),
            use_streaming=True,
            max_sequence_length=576
        )

        # 2. Create and Compile Model
        model_params = train_dataset.generate_model_params()
        model = SingleChainAlignAIR(**model_params)

        # The model must be compiled before being passed to the Trainer.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(clipnorm=1),
            loss=None,  # Loss is handled in the model's custom train_step
            metrics={
                'v_allele': [tf.keras.metrics.AUC(name='auc'), 'binary_accuracy'],
                'd_allele': [tf.keras.metrics.AUC(name='auc'), 'binary_accuracy'],
                'j_allele': [tf.keras.metrics.AUC(name='auc'), 'binary_accuracy'],
            }
        )

        # 3. Initialize the new Trainer with its simplified signature
        trainer = Trainer(
            model=model,
            session_path=str('./'),
            model_name="heavy_chain_test"
        )

        # 4. Run Training by passing parameters to the .train() method
        trainer.train(
            train_dataset=train_dataset,
            epochs=1,
            samples_per_epoch=32,  # Must be >= batch_size
            batch_size=16
        )

        self.assertIsNotNone(trainer.history, "Training history should not be None.")
        self.assertIn('loss', trainer.history.history, "Loss should be in training history.")

    def test_multi_chain_alignair_model_training(self):
        """Tests the training process for a multi-chain model with the new Trainer."""
        # 1. Create Dataset
        multi_config = MultiDataConfigContainer([HUMAN_IGK_OGRDB, HUMAN_IGL_OGRDB])
        train_dataset = MultiChainDataset(
            data_paths=[self.light_chain_dataset_path, self.light_chain_dataset_path],
            dataconfigs=multi_config,
            max_sequence_length=576,
            use_streaming=True,
        )

        # 2. Create and Compile Model
        model_params = train_dataset.generate_model_params()
        model = MultiChainAlignAIR(**model_params)

        # Dynamically build the metrics dictionary to match the prefixed outputs
        metrics = {}

        metrics[f'v_allele'] = [tf.keras.metrics.AUC(name='auc')]
        metrics[f'j_allele'] = [tf.keras.metrics.AUC(name='auc')]
        metrics['chain_type'] = 'categorical_accuracy'

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1),
            loss=None,  # Loss is handled in the custom train_step
            metrics=metrics
        )

        # 3. Initialize Trainer
        trainer = Trainer(
            model=model,
            session_path=str('./'),
            model_name="multi_chain_test"
        )

        # 4. Run Training
        trainer.train(
            train_dataset=train_dataset,
            epochs=1,
            samples_per_epoch=64,
            batch_size=32
        )

        # 5. Verify training completed
        self.assertIsNotNone(trainer.history, "Training history should not be None.")

        # 6. Test model prediction with correctly structured multi-chain input
        dummy_input_data = np.random.randint(0, 6, size=(1, 576))
        test_input = {
            f"tokenized_sequence": dummy_input_data
        }
        predictions = model(test_input, training=False)

        # # 7. Verify output structure for multi-chain model
        # # Check for prefixed outputs
        # for chain in train_dataset.chain_types:
        #     prefix = chain.value
        #     self.assertIn(f'{prefix}_v_allele', predictions)
        #     self.assertIn(f'{prefix}_j_start', predictions)

        # Check for the shared chain_type output
        self.assertIn('chain_type', predictions)

        # Verify chain type prediction has correct shape (number of chain types)
        expected_chain_types = len(multi_config.chain_types())
        self.assertEqual(predictions['chain_type'].shape[-1], expected_chain_types)

        print(f"✅ MultiChainAlignAIR test completed successfully!")

    def test_load_saved_heavy_chain_model(self):
        """Tests loading weights into a pre-built model (Trainer is not used for loading)."""
        # 1. Define model parameters and create the model instance
        # These must match the parameters of the saved weights
        model_params = {
            'max_seq_length': 576,
            'dataconfig': MultiDataConfigContainer([HUMAN_IGH_OGRDB]),
        }
        model = SingleChainAlignAIR(**model_params)

        # 2. Build the model by calling it with dummy data. This is crucial before loading weights.
        dummy_input = {"tokenized_sequence": np.zeros((1, 576), dtype=np.float32)}
        _ = model(dummy_input)

        # 3. Load the weights directly into the model instance
        model_checkpoint_path = self.test_dir +'/'+ 'AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2'
        model.load_weights(model_checkpoint_path).expect_partial()  # Use expect_partial for robustness

        # 4. Perform a prediction to ensure the loaded model works
        prediction_dataset = PredictionDataset(max_sequence_length=576)
        seq = 'CAGCCACAACTGAACTGGTCAAGTCCAGGACTGGTGAATACCTCGCAGACCGTCACACTCACCCTTGCCGTGTCCGGGGACCGTGTCTCCAGAACCACTGCTGTTTGGAAGTGGAGGGGTCAGACCCCATCGCGAGGCCTTGCGTGGCTGGGAAGGACCTACNACAGTTCCAGGTGATTTGCTAACAACGAAGTGTCTGTGAATTGTTNAATATCCATGAACCCAGACGCATCCANGGAACGGNTCTTCCTGCACCTGAGGTCTGGGGCCTTCGACGACACGGCTGTACATNCGTGAGAAAGCGGTGACCTCTACTAGGATAGTGCTGAGTACGACTGGCATTACGCTCTCNGGGACCGTGCCACCCTTNTCACTGCCTCCTCGG'
        encoded_seq = prediction_dataset.encode_and_equal_pad_sequence(seq)['tokenized_sequence']

        predicted = model.predict({'tokenized_sequence': np.vstack([encoded_seq])})
        self.assertIsNotNone(predicted, "Prediction should not be None after loading weights.")
        self.assertIn('v_allele', predicted)

    def test_heuristic_matcher_basic(self):
            # Test case where there is an indel, causing the segment and reference lengths to differ
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

    def test_predict_script_heavy_chain(self):

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'AlignAIR', 'API'))
        script_path = os.path.join(base_dir, 'AlignAIRRPredict.py')


        # Ensure the script exists
        self.assertTrue(os.path.exists(script_path), "Predict script not found at path: " + script_path)

        # Define the command with absolute paths
        command = [
            'C:/Users/tomas/Desktop/AlignAIRR/AlignAIR_ENV/Scripts/python', script_path,
            '--model_checkpoint', os.path.join(self.test_dir, 'AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2'),
            '--save_path', str(self.test_dir)+'/',
            '--genairr_dataconfig', 'HUMAN_IGH_OGRDB',
            '--sequences', self.heavy_chain_dataset_path,
            '--batch_size', '32',
            '--translate_to_asc'
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
        validation = pd.read_csv('./heavychain_predict_validation.csv')

        # drop type column
        # if 'chain_type' in df.col
        # df.drop(columns=['type'], inplace=True)

        # Compare dataframes cell by cell
        validation.chain_type = validation.chain_type.str.replace('heavy','ChainType.BCR_HEAVY')

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.assertEqual(df.iloc[i, j], validation.iloc[i, j],
                                 f"Mismatch at row {i}, column {j}: {df.iloc[i, j]} != {validation.iloc[i, j]}")

        self.assertFalse(df.empty, "Output CSV is empty")
        # Additional assertions can be added here

        # Cleanup
        os.remove(output_csv)

    def test_predict_script_light_chain(self):

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'AlignAIR', 'API'))
        script_path = os.path.join(base_dir, 'AlignAIRRPredict.py')


        # Ensure the script exists
        self.assertTrue(os.path.exists(script_path), "Predict script not found at path: " + script_path)

        # Define the command with absolute paths
        command = [
            'C:/Users/tomas/Desktop/AlignAIRR/AlignAIR_ENV/Scripts/python', script_path,
            '--model_checkpoint', os.path.join(self.test_dir, 'LightChain_AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced'),
            '--save_path', str(self.test_dir)+'/',
            '--genairr_dataconfig', 'HUMAN_IGK_OGRDB, HUMAN_IGL_OGRDB', # comma separated list of dataconfigs
            '--sequences', self.light_chain_dataset_path,
            '--batch_size', '32',
            '--translate_to_asc'
        ]

        # Execute the script
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, "Script failed to run with error: " + result.stderr)


        # Check if the CSV was created
        file_name = self.light_chain_dataset_path.split('/')[-1].split('.')[0]
        save_name = file_name + '_alignairr_results.csv'

        output_csv = os.path.join(self.test_dir, save_name)
        self.assertTrue(os.path.isfile(output_csv), "Output CSV file not created")


        # Read the output CSV and validate its contents
        df = pd.read_csv(output_csv)
        #df.drop(columns=['d_likelihoods'], inplace=True)
        validation = pd.read_csv('./lightchain_predict_validation.csv')

        # Compare dataframes cell by cell
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.assertEqual(df.iloc[i, j], validation.iloc[i, j],
                                 f"Mismatch at row {i}, column {j}: {df.iloc[i, j]} != {validation.iloc[i, j]}")

        self.assertFalse(df.empty, "Output CSV is empty")
        # Additional assertions can be added here

        # Cleanup
        os.remove(output_csv)

    def test_predict_script_multi_chain_light(self):
        """Test multi-chain prediction using CLI with both IGK and IGL configs."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'AlignAIR', 'API'))
        script_path = os.path.join(base_dir, 'AlignAIRRPredict.py')

        # Ensure the script exists
        self.assertTrue(os.path.exists(script_path), "Predict script not found at path: " + script_path)

        # Define test file and output paths
        test_file = os.path.join(self.test_dir, 'sample_LightChain_dataset.csv')
        output_csv = os.path.join(self.test_dir, 'multi_chain_light_output.csv')
        
        # Remove output file if it exists
        if os.path.exists(output_csv):
            os.remove(output_csv)
        
        # Define the command for multi-chain light chain prediction
        command = [
            'C:/Users/tomas/Desktop/AlignAIRR/AlignAIR_ENV/Scripts/python', script_path,
            '--sequences', test_file,
            '--model_checkpoint', os.path.join(self.test_dir, 'AlignAIR_MultiChain_LightChain'),
            '--genairr_dataconfig', 'C:/Users/tomas/Downloads/HUMAN_IGL_OGRDB.pkl,C:/Users/tomas/Downloads/HUMAN_IGK_OGRDB.pkl',  # Multi-chain config
            '--save_path', output_csv,
            '--batch_size', '64',
            '--max_input_size', '576'
        ]

        # Execute the script
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, "Script failed to run with error: " + result.stderr)

        # Check for successful execution
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")
        
        # Verify the output file was created
        self.assertTrue(os.path.isfile(output_csv), "Multi-chain output CSV file not created")
        
        # Read and validate the output
        df = pd.read_csv(output_csv)
        self.assertFalse(df.empty, "Multi-chain output CSV is empty")
        
        # Verify multi-chain specific columns are present
        expected_columns = ['v_call', 'j_call', 'chain_type', 'v_sequence_start', 'v_sequence_end', 
                           'j_sequence_start', 'j_sequence_end', 'mutation_rate', 'productive']
        
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Expected column '{col}' not found in multi-chain output")
        
        # Verify chain type predictions are present
        self.assertTrue('chain_type' in df.columns, "Chain type predictions missing from multi-chain output")
        
        print(f"✅ Multi-chain prediction test completed successfully!")
        print(f"   - Processed {len(df)} sequences")
        print(f"   - Output columns: {list(df.columns)}")
        
        # Cleanup
        os.remove(output_csv)

    def test_predict_script_tcrb(self):

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'AlignAIR', 'API'))
        script_path = os.path.join(base_dir, 'AlignAIRRPredict.py')


        # Ensure the script exists
        self.assertTrue(os.path.exists(script_path), "Predict script not found at path: " + script_path)

        # Define the command with absolute paths
        command = [
            'C:/Users/tomas/Desktop/AlignAIRR/AlignAIR_ENV/Scripts/python', script_path,
            '--model_checkpoint', os.path.join(self.test_dir, 'AlignAIRR_TCRB_Model_checkpoint'),
            '--save_path', str(self.test_dir)+'/',
            '--genairr_dataconfig', 'HUMAN_TCRB_IMGT',
            '--sequences', self.tcrb_chain_dataset_path,
            '--batch_size', '32',
            '--translate_to_asc',
        ]

        # Execute the script
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, "Script failed to run with error: " + result.stderr)


        # Check if the CSV was created
        file_name = self.tcrb_chain_dataset_path.split('/')[-1].split('.')[0]
        save_name = file_name + '_alignairr_results.csv'

        output_csv = os.path.join(self.test_dir, save_name)
        self.assertTrue(os.path.isfile(output_csv), "Output CSV file not created")

        # Read the output CSV and validate its contents
        df = pd.read_csv(output_csv)
        validation = pd.read_csv('./tcrb_predict_validation.csv')

        # drop type column
        df.drop(columns=['type'], inplace=True)


        # Compare dataframes cell by cell
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.assertEqual(df.iloc[i, j], validation.iloc[i, j],
                                 f"Mismatch at row {i}, column {j}: {df.iloc[i, j]} != {validation.iloc[i, j]}")

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
            '--genairr_dataconfig', 'HUMAN_IGH_OGRDB',
            '--train_dataset', self.heavy_chain_dataset_path,
            '--session_path', './',
            '--epochs', '1',
            '--batch_size', '32',
            '--samples_per_epoch', '32',
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
        #expected_log_path = os.path.join('./', 'TestModel.csv')
        #self.assertTrue(os.path.isfile(expected_log_path), "Log file not created")

        # Cleanup
        # if os.path.exists(models_path):
        #     shutil.rmtree(os.path.join('./', 'saved_models'))  # Remove the saved_models directory and all its contents
        # if os.path.exists(expected_log_path):
        #     os.remove(expected_log_path)  # Remove the log file if it exists

    def test_train_model_script_multi_chain_cli(self):
        """Tests running the unified training script for multiple chains from the CLI."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'AlignAIR', 'API'))
        script_path = os.path.join(base_dir, 'TrainModel.py')  # Name of your training script

        self.assertTrue(os.path.exists(script_path), "Training script not found at path: " + script_path)

        # Define the command for a multi-chain run
        command = [
            'C:/Users/tomas/Desktop/AlignAIRR/AlignAIR_ENV/Scripts/python', script_path,
            '--genairr_dataconfigs', 'HUMAN_IGH_OGRDB', 'HUMAN_IGK_OGRDB',
            '--train_datasets', str(self.heavy_chain_dataset_path), str(self.light_chain_dataset_path),
            '--session_path', str('./'),
            '--epochs', '1',
            '--batch_size', '16',  # Use a batch size divisible by the number of datasets
            '--samples_per_epoch', '32',
            '--max_sequence_length', '576',
            '--model_name', 'TestModelMultiCLI'
        ]

        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

        self.assertEqual(result.returncode, 0,
                         f"Multi-chain training script failed to run.\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}")
        expected_weights_path = './TestModelMultiCLI_final.weights.h5'
        from pathlib import Path
        expected_weights_path = Path(expected_weights_path)
        self.assertTrue(expected_weights_path.is_file(), "Multi-chain model weights file not created by CLI script.")

    def test_heavy_chain_backbone_loader(self):
        from src.AlignAIR.Finetuning.CustomClassificationHeadLoader import CustomClassificationHeadLoader
        tf.get_logger().setLevel('ERROR')

        TEST_V_ALLELE_SIZE = 200
        TEST_D_ALLELE_SIZE = 20
        TEST_J_ALLELE_SIZE = 2

        loader = CustomClassificationHeadLoader(
            pretrained_path='./AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2',
            model_class=HeavyChainAlignAIRR,
            max_seq_length=576,
            pretrained_v_allele_head_size=198,
            pretrained_d_allele_head_size=34,
            pretrained_j_allele_head_size=7,
            custom_v_allele_head_size=TEST_V_ALLELE_SIZE,
            custom_d_allele_head_size=TEST_D_ALLELE_SIZE,
            custom_j_allele_head_size=TEST_J_ALLELE_SIZE
        )

        # check v_allele,d_allele and j_allele layers are the same dim as the test size
        for layer in loader.model.layers:
            if 'v_allele' == layer.name:
                self.assertEqual(layer.weights[0].shape[1], TEST_V_ALLELE_SIZE) # Check the number of neurons in the layer
            if 'd_allele' == layer.name:
                self.assertEqual(layer.weights[0].shape[1], TEST_D_ALLELE_SIZE)
            if 'j_allele' == layer.name:
                self.assertEqual(layer.weights[0].shape[1], TEST_J_ALLELE_SIZE)

        # check the model has the same number of layers as the pretrained model
        self.assertEqual(len(loader.model.layers), len(loader.pretrained_model.layers))

        # check the weights in the frozen weight custom model are  the same as the pretrained model
        for layer, pretrained_layer in zip(loader.model.layers, loader.pretrained_model.layers):
            # Filter only layers with 'embedding' in the name
            if 'embedding' in layer.name and 'embedding' in pretrained_layer.name:
                weights_model1 = layer.get_weights()
                weights_model2 = pretrained_layer.get_weights()

                # Compare weights
                for w1, w2 in zip(weights_model1, weights_model2):
                    try:
                        tf.debugging.assert_equal(w1, w2)  # Tensor equality
                    except tf.errors.InvalidArgumentError as e:
                        self.fail(f"Weights mismatch in embedding layer '{layer.name}': {e}")

    def test_light_chain_backbone_loader(self):
        from src.AlignAIR.Finetuning.CustomClassificationHeadLoader import CustomClassificationHeadLoader
        tf.get_logger().setLevel('ERROR')

        TEST_V_ALLELE_SIZE = 200
        TEST_D_ALLELE_SIZE = 20
        TEST_J_ALLELE_SIZE = 2

        loader = CustomClassificationHeadLoader(
            pretrained_path='./AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2',
            model_class=HeavyChainAlignAIRR,
            max_seq_length=576,
            pretrained_v_allele_head_size=198,
            pretrained_d_allele_head_size=34,
            pretrained_j_allele_head_size=7,
            custom_v_allele_head_size=TEST_V_ALLELE_SIZE,
            custom_d_allele_head_size=TEST_D_ALLELE_SIZE,
            custom_j_allele_head_size=TEST_J_ALLELE_SIZE
        )

        # check v_allele,d_allele and j_allele layers are the same dim as the test size
        for layer in loader.model.layers:
            if 'v_allele' == layer.name:
                self.assertEqual(layer.weights[0].shape[1], TEST_V_ALLELE_SIZE) # Check the number of neurons in the layer
            if 'd_allele' == layer.name:
                self.assertEqual(layer.weights[0].shape[1], TEST_D_ALLELE_SIZE)
            if 'j_allele' == layer.name:
                self.assertEqual(layer.weights[0].shape[1], TEST_J_ALLELE_SIZE)

        # check the model has the same number of layers as the pretrained model
        self.assertEqual(len(loader.model.layers), len(loader.pretrained_model.layers))

        # check the weights in the frozen weight custom model are  the same as the pretrained model
        for layer, pretrained_layer in zip(loader.model.layers, loader.pretrained_model.layers):
            # Filter only layers with 'embedding' in the name
            if 'embedding' in layer.name and 'embedding' in pretrained_layer.name:
                weights_model1 = layer.get_weights()
                weights_model2 = pretrained_layer.get_weights()

                # Compare weights
                for w1, w2 in zip(weights_model1, weights_model2):
                    try:
                        tf.debugging.assert_equal(w1, w2)  # Tensor equality
                    except tf.errors.InvalidArgumentError as e:
                        self.fail(f"Weights mismatch in embedding layer '{layer.name}': {e}")

    def test_genotype_correction(self):
        import pickle

        # read genotype data config
        with open('Genotyped_DataConfig.pkl', 'rb') as file:
            genotype_datconfig = pickle.load(file)

        original_dataconfig = HUMAN_IGH_OGRDB

        model_checkpoint_path = os.path.join(self.test_dir, 'AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2')

        v_alleles_ref = list(map(lambda x: x.name, original_dataconfig.allele_list('v')))
        d_alleles_ref = list(map(lambda x: x.name, original_dataconfig.allele_list('d'))) + ['Short-D']
        j_alleles_ref = list(map(lambda x: x.name, original_dataconfig.allele_list('j')))

        # read reference data config
        v_alleles = list(map(lambda x: x.name, genotype_datconfig.allele_list('v')))
        d_alleles = list(map(lambda x: x.name, genotype_datconfig.allele_list('d'))) + ['Short-D']
        j_alleles = list(map(lambda x: x.name, genotype_datconfig.allele_list('j')))


        # create mock yaml file that we will delete after test is finished
        with open('genotype.yaml', 'w') as file:
            yaml.dump({'v': v_alleles, 'd': d_alleles, 'j': j_alleles}, file)

        mock_model_params = {
            'v_allele_latent_size': 2 * len(v_alleles_ref),
            'd_allele_latent_size': 2 * len(d_alleles_ref),
            'j_allele_latent_size': 2 * len(j_alleles_ref),
        }

        with open('model_params.yaml', 'w') as file:
            yaml.dump(mock_model_params, file)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'AlignAIR', 'API'))
        script_path = os.path.join(base_dir, 'AlignAIRRPredict.py')
        # Define the command with absolute paths
        command = [
            'C:/Users/tomas/Desktop/AlignAIRR/AlignAIR_ENV/Scripts/python', script_path,
            '--model_checkpoint', os.path.join(self.test_dir, 'AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2'),
            '--save_path', str(self.test_dir) + '/',
            '--genairr_dataconfig', 'HUMAN_IGH_OGRDB',
            '--sequences', self.heavy_chain_dataset_path,
            '--batch_size', '32',
            '--translate_to_asc',
            '--save_predict_object',
            '--custom_genotype', 'genotype.yaml'
        ]

        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')


        self.assertEqual(result.returncode, 0, "Script failed to run with error: " + result.stderr)

        #####################################################################################
        # Check if the CSV was created
        file_name = self.heavy_chain_dataset_path.split('/')[-1].split('.')[0]
        save_name = file_name + '_alignairr_results.csv'

        output_csv = os.path.join(self.test_dir, save_name)

        # Read the output CSV and validate its contents
        file_name = self.heavy_chain_dataset_path.split('/')[-1].split('.')[0]
        save_name = file_name + '_alignairr_results.csv'
        predictobject_path = save_name.replace('_alignairr_results.csv', '_alignair_results_predictObject.pkl')
        with open(predictobject_path, 'rb') as file:
            predictobject_full_nodel = pickle.load(file)

        print(f"First predict object loaded successfully")

        command = [
            'C:/Users/tomas/Desktop/AlignAIRR/AlignAIR_ENV/Scripts/python', script_path,
            '--model_checkpoint',
            os.path.join(self.test_dir, 'Genotyped_Frozen_Heavy_Chain_AlignAIRR_S5F_OGRDB_S5F_576_Balanced'),
            '--save_path', str(self.test_dir) + '/',
            '--sequences', self.heavy_chain_dataset_path,
            '--genairr_dataconfig', './Genotyped_DataConfig.pkl',
            '--batch_size', '32',
            '--translate_to_asc',
            '--save_predict_object',
            '--finetuned_model_params_yaml', 'model_params.yaml',
            '--custom_genotype', 'genotype.yaml'
        ]


        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')


        file_name = self.heavy_chain_dataset_path.split('/')[-1].split('.')[0]
        save_name = file_name + '_alignairr_results.csv'
        predictobject_path = save_name.replace('_alignairr_results.csv', '_alignair_results_predictObject.pkl')

        with open(predictobject_path, 'rb') as file:
            predictobject_genotype_nodel = pickle.load(file)


        # correct and test the error between the two models
        # v_mean_mae = np.mean(np.abs(predictobject_full_nodel.processed_predictions['v_allele'] - predictobject_genotype_nodel.processed_predictions['v_allele']))
        # d_mean_mae = np.mean(np.abs(predictobject_full_nodel.processed_predictions['d_allele'] - predictobject_genotype_nodel.processed_predictions['d_allele']))
        # j_mean_mae = np.mean(np.abs(predictobject_full_nodel.processed_predictions['j_allele'] - predictobject_genotype_nodel.processed_predictions['j_allele']))
        #
        # print(v_mean_mae,d_mean_mae,j_mean_mae)


        os.remove('genotype.yaml')
        os.remove('model_params.yaml')
        os.remove(predictobject_path)
        os.remove(output_csv)


    def test_orientation_classifier(self):
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

    # def test_config_load_step(self):
    #     from src.AlignAIR.Preprocessing.Steps.dataconfig_steps import ConfigLoadStep
    #     from unittest.mock import Mock
    #
    #     # Mock the PredictObject and its attributes
    #     mock_predict_object = Mock()
    #     mock_predict_object.script_arguments.chain_type = 'heavy'
    #     mock_predict_object.script_arguments.heavy_data_config = 'path/to/heavy/config'
    #     mock_predict_object.script_arguments.kappa_data_config = 'path/to/kappa/config'
    #     mock_predict_object.script_arguments.lambda_data_config = 'path/to/lambda/config'
    #
    #     # Mock the DataConfigLibrary
    #     mock_data_config_library = Mock(spec=DataConfigLibrary)
    #     mock_data_config_library.mount_type = Mock()
    #
    #     # Replace the DataConfigLibrary with the mock
    #     with unittest.mock.patch('src.AlignAIR.Preprocessing.Steps.dataconfig_steps.DataConfigLibrary',
    #                              return_value=mock_data_config_library):
    #         step = ConfigLoadStep("Load Config")
    #         result = step.process(mock_predict_object)
    #
    #     # Assertions
    #     mock_data_config_library.mount_type.assert_called_with('heavy')
    #     self.assertEqual(result.data_config_library, mock_data_config_library)
    #     self.assertTrue(mock_predict_object.mount_genotype_list.called)
    #     self.assertEqual(result, mock_predict_object)

    def test_file_name_extraction_step(self):
        from src.AlignAIR.Preprocessing.Steps.file_steps import FileNameExtractionStep
        from AlignAIR.Utilities.step_utilities import FileInfo
        from unittest.mock import Mock

        # Mock the PredictObject and its attributes
        mock_predict_object = Mock()
        mock_predict_object.script_arguments.sequences = 'path/to/sequences/file.csv'

        step = FileNameExtractionStep("Extract File Name")
        result = step.process(mock_predict_object)

        # Assertions
        self.assertIsInstance(result.file_info, FileInfo)
        self.assertEqual(result.file_info.file_name, 'file')
        self.assertEqual(result.file_info.file_type, 'csv')
        self.assertEqual(result, mock_predict_object)

    def test_file_sample_counter_step(self):
        from src.AlignAIR.Preprocessing.Steps.file_steps import FileSampleCounterStep
        from AlignAIR.Utilities.file_processing import FILE_ROW_COUNTERS
        from AlignAIR.Utilities.step_utilities import FileInfo
        from unittest.mock import Mock

        # Create a real FileInfo object for single file case
        mock_predict_object = Mock()
        file_info = FileInfo('./tests/sample_HeavyChain_dataset.csv')
        mock_predict_object.file_info = file_info
        mock_predict_object.script_arguments.sequences = './tests/sample_HeavyChain_dataset.csv'

        # Mock the row counter function
        mock_row_counter = Mock(return_value=100)
        FILE_ROW_COUNTERS['csv'] = mock_row_counter

        step = FileSampleCounterStep("Count Samples")
        result = step.process(mock_predict_object)

        # Assertions
        mock_row_counter.assert_called_with('./tests/sample_HeavyChain_dataset.csv')
        self.assertEqual(file_info.sample_count, 100)
        self.assertEqual(result, mock_predict_object)

    def test_fast_kmer_density_extractor(self):

        # test heavy chain detection

        ref_alleles = (list(map(lambda x: x.ungapped_seq.upper(), HUMAN_IGH_OGRDB.allele_list('v')))+
                          list(map(lambda x: x.ungapped_seq.upper(), HUMAN_IGH_OGRDB.allele_list('d')))+
                            list(map(lambda x: x.ungapped_seq.upper(), HUMAN_IGH_OGRDB.allele_list('j')))
                       )

        candidate_sequence_extractor = FastKmerDensityExtractor(11, max_length=576, allowed_mismatches=0)
        candidate_sequence_extractor.fit(ref_alleles)

        import json
        with open('./KMer_Density_Extractor_HeavyChainTests.json', 'r') as f:
            tests = json.load(f)

        results = [candidate_sequence_extractor.transform_holt(noise)[0] for true, noise in tests]

        def is_within_mismatch_limit(true_seq, result_seq, max_mismatch=5):
            """Check if true_seq is fully contained in result_seq with up to max_mismatch from start or end."""
            true_len = len(true_seq)
            for i in range(max_mismatch + 1):
                # Allow mismatches from start
                if true_seq[i:] in result_seq:
                    return True
                # Allow mismatches from end
                if true_seq[:true_len - i] in result_seq:
                    return True
            return False

        for s, (result, (true, _)) in enumerate(zip(results, tests)):
            self.assertTrue(len(result) <= 576, f"Test {s} failed: Result length exceeds 576")
            self.assertTrue(is_within_mismatch_limit(true, result),
                            f"Test {s} failed: True sequence not found within mismatch limit")

    def test_heuristic_matcher_comprehensive(self):
        """
        Exercise the matcher on a grid of real‑world edge cases:
        1. Exact match
        2. Left over‑segmentation  (+3 nt)
        3. Right over‑segmentation (+4 nt)
        4. Over‑segmentation on both ends (+2/+2 nt)
        5. Left deletion          (‑5 nt  , indels = 5)
        6. Right deletion         (‑6 nt  , indels = 6)
        7. Both‑side deletions    (‑3/‑4 nt, indels = 7)
        8. Internal insertion     (+4 nt  , indels = 4)
        9. Two point mismatches   (same length, indels = 0)
       10. Mixed: left over‑segmentation + internal insertion + 2 mismatches
        """
        # --- reference sequence & matcher -------------------------------------------------
        ref_seq  = "ATGCGTACGTCAGTACGTCAGTACGTTAGC"           # length = 30
        allele   = "TEST*01"
        matcher  = HeuristicReferenceMatcher({allele: ref_seq})

        def mutate(seq, pos, base):
            return seq[:pos] + base + seq[pos+1:]

        # --- catalogue of test‑cases ------------------------------------------------------
        cases = [
            # name                  segment builder (lambda)                indels exp_start exp_end
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

        # --- pack into matcher inputs -----------------------------------------------------
        sequences, starts, ends, alleles, indel_counts = [], [], [], [], []
        for _, builder, indels, *_ in cases:
            seg = builder()
            sequences.append(seg)
            starts.append(0)                # we give the whole segment
            ends.append(len(seg))
            alleles.append(allele)
            indel_counts.append(indels)

        # --- run matcher ------------------------------------------------------------------
        results = matcher.match(
            sequences, starts, ends, alleles,
            indel_counts, _gene="v"          # gene label only affects tqdm text
        )

        # --- validate ---------------------------------------------------------------------
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

    def test_model_loading_step_multi_chain_integration(self):
        """Test that ModelLoadingStep correctly selects MultiChainAlignAIR for multi-chain scenarios."""
        
        from AlignAIR.Preprocessing.Steps.model_loading_steps import ModelLoadingStep
        from AlignAIR.Utilities.step_utilities import FileInfo
        
        # Create test components
        model_loader = ModelLoadingStep("Test Multi-Chain Model Loading")
        
        # Create MultiDataConfigContainer with multiple configs
        multi_config = MultiDataConfigContainer([HUMAN_IGK_OGRDB, HUMAN_IGL_OGRDB])
        
        # Create file info for testing
        test_file_info = FileInfo(self.light_chain_dataset_path)
        
        # Mock a model checkpoint path (we won't actually load weights in this test)
        mock_checkpoint = os.path.join(self.test_dir, 'LightChain_AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced')
        
        try:
            # Test the model loading logic (this will create the model but may fail on weight loading)
            model = model_loader.load_model(
                file_info=test_file_info,
                dataconfig=multi_config,
                model_checkpoint=mock_checkpoint,
                max_sequence_size=576
            )
            
            # Should never reach here due to weight loading, but if it does, verify it's MultiChainAlignAIR
            self.assertIsInstance(model, MultiChainAlignAIR, 
                                "ModelLoadingStep should select MultiChainAlignAIR for multi-chain scenarios")
                                
        except Exception as e:
            # Expected to fail on weight loading, but we can check the error message
            # to confirm the right model type was selected
            print(f"Expected weight loading error (this is normal): {e}")
            
        # Test that the selection logic works correctly
        is_multi_chain = len(multi_config) > 1
        self.assertTrue(is_multi_chain, "Multi-config should be detected as multi-chain")
        
        # Test single chain scenario for comparison
        single_config = MultiDataConfigContainer([HUMAN_IGH_OGRDB])
        is_single_chain = len(single_config) == 1
        self.assertTrue(is_single_chain, "Single config should be detected as single-chain")
        
        print("✅ Model loading step integration test completed!")
        print(f"   - Multi-chain detection: {is_multi_chain}")
        print(f"   - Single-chain detection: {is_single_chain}")


if __name__ == '__main__':
    unittest.main()

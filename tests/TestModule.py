import os

import yaml

from AlignAIR.Data.PredictionDataset import PredictionDataset
from src.AlignAIR.Metadata import RandomDataConfigGenerator
from src.AlignAIR.Models.LightChain import LightChainAlignAIRR
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

        model_parmas = train_dataset.generate_model_params()
        model = HeavyChainAlignAIRR(**model_parmas)

        model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1),
                      loss = None,
                      metrics={
                            'v_start':tf.keras.losses.mse,
                            'v_end':tf.keras.losses.mse,
                            'd_start':tf.keras.losses.mse,
                            'd_end':tf.keras.losses.mse,
                            'j_start':tf.keras.losses.mse,
                            'j_end':tf.keras.losses.mse,
                            'v_allele':tf.keras.losses.binary_crossentropy,
                            'd_allele':tf.keras.losses.binary_crossentropy,
                            'j_allele':tf.keras.losses.binary_crossentropy,
                      }
                      )



        trainer = Trainer(
            model=model,
            batch_size=256,
            epochs=1,
            steps_per_epoch=4,
            verbose=1,
            classification_metric=[tf.keras.metrics.AUC(), tf.keras.metrics.AUC(), tf.keras.metrics.AUC()],
            regression_metric=tf.keras.losses.binary_crossentropy,
        )

        # Train the model
        trainer.train(train_dataset)



        self.assertIsNotNone(trainer.history)

    def test_light_chain_model_training(self):


        train_dataset = LightChainDataset(data_path=self.light_chain_dataset_path,
                                          lambda_dataconfig=builtin_lambda_chain_data_config(),
                                          kappa_dataconfig=builtin_kappa_chain_data_config(),
                                          batch_read_file=True,
                                          max_sequence_length=576)

        model_parmas = train_dataset.generate_model_params()

        model = LightChainAlignAIRR(**model_parmas)

        model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1),
                      loss=None,
                      metrics={
                          'v_start': tf.keras.losses.mse,
                          'v_end': tf.keras.losses.mse,
                          'j_start': tf.keras.losses.mse,
                          'j_end': tf.keras.losses.mse,
                          'v_allele': tf.keras.losses.binary_crossentropy,
                          'j_allele': tf.keras.losses.binary_crossentropy,
                      }
                      )

        trainer = Trainer(
            model=model,
            batch_size=256,
            epochs=1,
            steps_per_epoch=4,
            verbose=1,
            classification_metric=[tf.keras.metrics.AUC(), tf.keras.metrics.AUC(), tf.keras.metrics.AUC()],
            regression_metric=tf.keras.losses.binary_crossentropy,
        )

        # Train the model
        trainer.train(train_dataset)

        self.assertIsNotNone(trainer.history)

    def test_load_saved_heavy_chain_model(self):

        model_params = {'max_seq_length': 576, 'v_allele_count': 198, 'd_allele_count': 34, 'j_allele_count': 7}
        model = HeavyChainAlignAIRR(**model_params)
        trainer = Trainer(
            model=model,
            epochs=1,
            batch_size=32,
            steps_per_epoch=1,
            verbose=1,
        )
        MODEL_CHECKPOINT = './AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2'
        trainer.load_model(MODEL_CHECKPOINT,max_seq_length=model_params['max_seq_length'])

        self.assertNotEqual(trainer.model.log_var_v_end.weights[0].numpy(),0.0)

        prediction_Dataset = PredictionDataset(max_sequence_length=576)
        seq = 'CAGCCACAACTGAACTGGTCAAGTCCAGGACTGGTGAATACCTCGCAGACCGTCACACTCACCCTTGCCGTGTCCGGGGACCGTGTCTCCAGAACCACTGCTGTTTGGAAGTGGAGGGGTCAGACCCCATCGCGAGGCCTTGCGTGGCTGGGAAGGACCTACNACAGTTCCAGGTGATTTGCTAACAACGAAGTGTCTGTGAATTGTTNAATATCCATGAACCCAGACGCATCCANGGAACGGNTCTTCCTGCACCTGAGGTCTGGGGCCTTCGACGACACGGCTGTACATNCGTGAGAAAGCGGTGACCTCTACTAGGATAGTGCTGAGTACGACTGGCATTACGCTCTCNGGGACCGTGCCACCCTTNTCACTGCCTCCTCGG'
        es = prediction_Dataset.encode_and_equal_pad_sequence(seq)['tokenized_sequence']
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
            '--model_checkpoint', os.path.join(self.test_dir, 'AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2'),
            '--save_path', str(self.test_dir)+'/',
            '--chain_type', 'heavy',
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
        df.drop(columns=['d_likelihoods'], inplace=True)
        validation = pd.read_csv('./heavychain_predict_validation.csv',index_col=0)
        df_equal = df.equals(validation)
        self.assertTrue(df_equal, "Output CSV does not match the expected output")
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
        #expected_log_path = os.path.join('./', 'TestModel.csv')
        #self.assertTrue(os.path.isfile(expected_log_path), "Log file not created")

        # Cleanup
        # if os.path.exists(models_path):
        #     shutil.rmtree(os.path.join('./', 'saved_models'))  # Remove the saved_models directory and all its contents
        # if os.path.exists(expected_log_path):
        #     os.remove(expected_log_path)  # Remove the log file if it exists

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
        import  pickle
        # read genotype data config
        with open('Genotyped_DataConfig.pkl', 'rb') as file:
            genotype_datconfig = pickle.load(file)

        ref_dataconfig = builtin_heavy_chain_data_config()

        v_alleles = [i.name  for j in genotype_datconfig.v_alleles for i in genotype_datconfig.v_alleles[j]]
        d_alleles = [i.name  for j in genotype_datconfig.d_alleles for i in genotype_datconfig.d_alleles[j]]
        d_alleles += ['Short-D']
        j_alleles = [i.name  for j in genotype_datconfig.j_alleles for i in genotype_datconfig.j_alleles[j]]

        v_alleles_ref = [i.name for j in ref_dataconfig.v_alleles for i in ref_dataconfig.v_alleles[j]]
        d_alleles_ref = [i.name for j in ref_dataconfig.d_alleles for i in ref_dataconfig.d_alleles[j]]
        d_alleles_ref += ['Short-D']
        j_alleles_ref = [i.name for j in ref_dataconfig.j_alleles for i in ref_dataconfig.j_alleles[j]]

        # create mock yamal file tat we will delete after test is finished
        with open('genotype.yaml', 'w') as file:
            yaml.dump({'v':v_alleles,'d':d_alleles,'j':j_alleles}, file)

        mock_model_params = {'v_allele_latent_size':2*len(v_alleles_ref),
                                'd_allele_latent_size':2*len(d_alleles_ref),
                                'j_allele_latent_size':2*len(j_alleles_ref),
                             'v_allele_count': len(v_alleles), 'd_allele_count': len(d_alleles), 'j_allele_count': len(j_alleles)}

        with open('model_params.yaml', 'w') as file:
            yaml.dump(mock_model_params, file)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'AlignAIR', 'API'))
        script_path = os.path.join(base_dir, 'AlignAIRRPredict.py')

        # Ensure the script exists

        # Define the command with absolute paths
        command = [
            'C:/Users/tomas/Desktop/AlignAIRR/AlignAIR_ENV/Scripts/python', script_path,
            '--model_checkpoint', os.path.join(self.test_dir, 'AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2'),
            '--save_path', str(self.test_dir) + '/',
            '--chain_type', 'heavy',
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

        command = [
            'C:/Users/tomas/Desktop/AlignAIRR/AlignAIR_ENV/Scripts/python', script_path,
            '--model_checkpoint', os.path.join(self.test_dir, 'Genotyped_Frozen_Heavy_Chain_AlignAIRR_S5F_OGRDB_S5F_576_Balanced'),
            '--save_path', str(self.test_dir) + '/',
            '--chain_type', 'heavy',
            '--sequences', self.heavy_chain_dataset_path,
            '--heavy_data_config', 'Genotyped_DataConfig.pkl',
            '--batch_size', '32',
            '--translate_to_asc',
            '--save_predict_object',
            '--finetuned_model_params_yaml', 'model_params.yaml',
            '--custom_genotype', 'genotype.yaml'
        ]

        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, "Script failed to run with error: " + result.stderr)

        file_name = self.heavy_chain_dataset_path.split('/')[-1].split('.')[0]
        save_name = file_name + '_alignairr_results.csv'
        predictobject_path = save_name.replace('_alignairr_results.csv','_alignair_results_predictObject.pkl')

        with open(predictobject_path, 'rb') as file:
            predictobject_genotype_nodel = pickle.load(file)



        # correct and test the error between the two models

        v_mean_mae = np.mean(np.abs(predictobject_full_nodel.processed_predictions['v_allele'] - predictobject_genotype_nodel.processed_predictions['v_allele']))
        d_mean_mae = np.mean(np.abs(predictobject_full_nodel.processed_predictions['d_allele'] - predictobject_genotype_nodel.processed_predictions['d_allele']))
        j_mean_mae = np.mean(np.abs(predictobject_full_nodel.processed_predictions['j_allele'] - predictobject_genotype_nodel.processed_predictions['j_allele']))

        print(v_mean_mae,d_mean_mae,j_mean_mae)

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

        classifier = builtin_orientation_classifier('light')
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

    def test_config_load_step(self):
        from src.AlignAIR.Preprocessing.Steps.dataconfig_steps import ConfigLoadStep
        from AlignAIR.Utilities.step_utilities import DataConfigLibrary
        from unittest.mock import Mock

        # Mock the PredictObject and its attributes
        mock_predict_object = Mock()
        mock_predict_object.script_arguments.chain_type = 'heavy'
        mock_predict_object.script_arguments.heavy_data_config = 'path/to/heavy/config'
        mock_predict_object.script_arguments.kappa_data_config = 'path/to/kappa/config'
        mock_predict_object.script_arguments.lambda_data_config = 'path/to/lambda/config'

        # Mock the DataConfigLibrary
        mock_data_config_library = Mock(spec=DataConfigLibrary)
        mock_data_config_library.mount_type = Mock()

        # Replace the DataConfigLibrary with the mock
        with unittest.mock.patch('src.AlignAIR.Preprocessing.Steps.dataconfig_steps.DataConfigLibrary',
                                 return_value=mock_data_config_library):
            step = ConfigLoadStep("Load Config")
            result = step.process(mock_predict_object)

        # Assertions
        mock_data_config_library.mount_type.assert_called_with('heavy')
        self.assertEqual(result.data_config_library, mock_data_config_library)
        self.assertTrue(mock_predict_object.mount_genotype_list.called)
        self.assertEqual(result, mock_predict_object)

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
        from unittest.mock import Mock

        # Mock the PredictObject and its attributes
        mock_predict_object = Mock()
        mock_predict_object.file_info.file_type = 'csv'
        mock_predict_object.script_arguments.sequences = './sample_HeavyChain_dataset.csv'

        # Mock the row counter function
        mock_row_counter = Mock(return_value=100)
        FILE_ROW_COUNTERS['csv'] = mock_row_counter

        step = FileSampleCounterStep("Count Samples")
        result = step.process(mock_predict_object)

        # Assertions
        mock_row_counter.assert_called_with('./sample_HeavyChain_dataset.csv')
        self.assertEqual(result, mock_predict_object)







if __name__ == '__main__':
    unittest.main()

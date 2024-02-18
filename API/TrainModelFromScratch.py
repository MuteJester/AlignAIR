import subprocess
import pandas as pd
import argparse
import tensorflow as tf
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import random
import tensorflow as tf
import pandas as pd
from Data import HeavyChainDataset,LightChainDataset
from SequenceSimulation.sequence import LightChainSequence
import pickle
from Models.HeavyChain import HeavyChainAlignAIRR
from Models.LightChain import LightChainAlignAIRR
from Trainers import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlingAIRR Model Trainer from Scratch')
    #parser.add_argument('--parameter_csv_file', type=str, required=True, help='data config path for heavy chain or lambda chain')
    #args = parser.parse_args()

    #parameters_df = pd.read_csv(args.parameter_csv_file,encoding='utf8')
    parameters_df = pd.read_csv('E:\Immunobiology\AlignAIRR\API\MTFS_parameters.csv',encoding='utf8')
    # Convert parameters DataFrame to a dictionary
    parameters = parameters_df.set_index('parameter')['value'].to_dict()
    # Convert all non-string values to strings
    for key, value in parameters.items():
        if not isinstance(value, str):
            parameters[key] = str(value)

    # Generate Sequences
    # Define the command and parameters
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    generate_train_dataset_path = os.path.join(current_script_path, 'GenerateTrainDataset.py')

    # Construct the command
    command = [
        "python", "-m", "AlignAIRR.API.GenerateTrainDataset",
        "--dataconfig_path", parameters['data_config_heavy'] if parameters['chain_type'] == 'heavy' else parameters['data_config_lambda'],
        "--mutation_model", parameters['mutation_model'],
        "--save_path", parameters['data_save_path'],
        "--n_samples", str(parameters['number_of_train_samples']),
        "--chain_type", parameters['chain_type'],
        "--min_mutation_rate", str(parameters['min_mutation_rate']),
        "--max_mutation_rate", str(parameters['max_mutation_rate']),
        "--n_ratio", str(parameters['n_ratio']),
        "--max_sequence_length", str(parameters['max_sequence_length']),
        "--nucleotide_add_coefficient", str(parameters['nucleotide_add_coefficient']),
        "--nucleotide_remove_coefficient", str(parameters['nucleotide_remove_coefficient']),
        "--nucleotide_add_after_remove_coefficient", str(parameters['nucleotide_add_after_remove_coefficient']),
        "--random_sequence_add_proba", str(parameters['random_sequence_add_proba']),
        "--single_base_stream_proba", str(parameters['single_base_stream_proba']),
        "--duplicate_leading_proba", str(parameters['duplicate_leading_proba']),
        "--random_allele_proba", str(parameters['random_allele_proba']),
        "--corrupt_proba", str(parameters['corrupt_proba']),
        "--short_d_length", str(parameters['short_d_length']),
        "--save_mutations_record", str(parameters['save_mutations_record']),
        "--save_ns_record", str(parameters['save_ns_record'])
    ]

    # Include kappa and lambda dataconfig if they exist
    if 'data_config_kappa' in parameters and parameters['data_config_kappa']:
        command += ["--dataconfig_kappa", parameters['data_config_kappa']]


    # Run the command
    subprocess.run(command)

    # now that the above finished we have a dataset ready, we can start and train the model


    seed = 42
    np.random.seed(seed)  # Set the random seed
    tf.random.set_seed(seed)  # Set the random seed
    random.seed(seed)  # Set the random seed

    ### Start - CallBacks Definitions ###

    reduce_lr = ReduceLROnPlateau(
        monitor="v_allele_auc",
        factor=0.9,
        patience=20,
        min_delta=0.001,
        mode="auto",
    )
    # first start by reading in the dataconfigs so we can initiate a new model
    if parameters['chain_type'] == 'heavy':
        with open(parameters['data_config_heavy'], 'rb') as h:
            heavychain_config = pickle.load(h)

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=parameters['model_save_path'],
                save_weights_only=True,
                save_best_only=True,
                monitor="loss",
                mode="min",
            )

            TRAIN_DATASET = parameters['data_save_path']
            train_dataset = HeavyChainDataset(data_path=TRAIN_DATASET
                                              , dataconfig=heavychain_config,
                                              batch_size=int(parameters['training_batch_size']),
                                              max_sequence_length=int(parameters['max_sequence_length']),
                                              batch_read_file=True)

            print('Starting The Training...')
            print('Train Dataset Path: ', TRAIN_DATASET)
            trainer = Trainer(
                model=HeavyChainAlignAIRR,
                dataset=train_dataset,
                epochs=int(parameters['training_epochs']),
                steps_per_epoch=max(1,train_dataset.data_length//100),
                verbose=1,
                classification_metric=[tf.keras.metrics.AUC(), tf.keras.metrics.AUC(), tf.keras.metrics.AUC()],
                regression_metric=tf.keras.losses.binary_crossentropy,
                callbacks=[
                    reduce_lr,
                    model_checkpoint_callback,
                ],
                optimizers_params={"clipnorm": 1},
            )

            # Train the model
            trainer.train()


            trainer.model.save_weights(parameters['model_save_path'] + f'AlignAIRR_weights_final_epoch')
            trainer.save_model(parameters['model_save_path'] + f'AlignAIRR_weights_final_epoch')


    else:
        with open(parameters['data_config_lambda'], 'rb') as h:
            lambda_config = pickle.load(h)
        with open(parameters['data_config_kappa'], 'rb') as h:
            kappa_config = pickle.load(h)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=parameters['model_save_path'],
            save_weights_only=True,
            save_best_only=True,
            monitor="loss",
            mode="min",
        )

        TRAIN_DATASET = parameters['data_save_path']
        train_dataset = LightChainDataset(data_path=TRAIN_DATASET
                                          , lambda_dataconfig=lambda_config,
                                          kappa_dataconfig=kappa_config,
                                          batch_size=int(parameters['training_batch_size']),
                                          max_sequence_length=int(parameters['max_sequence_length']),
                                          batch_read_file=True)

        print('Starting The Training...')
        print('Train Dataset Path: ', TRAIN_DATASET)
        trainer = Trainer(
            model=LightChainAlignAIRR,
            dataset=train_dataset,
            epochs=int(parameters['training_epochs']),
            steps_per_epoch=max(1, train_dataset.data_length // 100),
            verbose=1,
            classification_metric=[tf.keras.metrics.AUC(), tf.keras.metrics.AUC(), tf.keras.metrics.AUC()],
            regression_metric=tf.keras.losses.binary_crossentropy,
            callbacks=[
                reduce_lr,
                model_checkpoint_callback,
            ],
            optimizers_params={"clipnorm": 1},
        )

        # Train the model
        trainer.train()

        trainer.model.save_weights(parameters['model_save_path'] + f'AlignAIRR_weights_final_epoch')
        trainer.save_model(parameters['model_save_path'] + f'AlignAIRR_weights_final_epoch')


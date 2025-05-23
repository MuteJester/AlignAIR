import argparse
import logging
import pickle
import numpy as np
from matplotlib import pyplot as plt
from scikitplot.metrics import plot_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
import random
from GenAIRR.simulation import HeavyChainSequenceAugmentor, LightChainSequenceAugmentor, SequenceAugmentorArguments, \
    LightChainKappaLambdaSequenceAugmentor
from GenAIRR.dataconfig import DataConfig
from GenAIRR.data import builtin_heavy_chain_data_config,builtin_lambda_chain_data_config,builtin_kappa_chain_data_config
from GenAIRR.mutation import Uniform

from AlignAIR.Preprocessing.Orientation import reverse_sequence, complement_sequence, reverse_complement_sequence

# Initialize DataConfig
data_config_builtin = builtin_heavy_chain_data_config()
kappa_data_config_builtin = builtin_kappa_chain_data_config()
lambda_data_config_builtin = builtin_lambda_chain_data_config()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='AlingAIR Model Prediction')
    parser.add_argument('--save_path', type=str, required=True, help='where to save the trained model and vectorizer')
    parser.add_argument('--plot_save_path', type=str, required=True, help='where to save the model evaluation plots')
    parser.add_argument('--chain_type', type=str, required=True, help='heavy / light')

    parser.add_argument('--train_dataset_size', type=int, default=100_000, help='number of train dataset samples (this is split evenly between uniform and s5f samples)')
    parser.add_argument('--uniform_test_dataset_size', type=int, default=100_000, help='number of uniform mutation test dataset samples')
    parser.add_argument('--partial_test_dataset_size', type=int, default=100_000, help='number of uniform mutation test dataset partial samples (only V/D/J / VJ / VD / DJ)')

    args = parser.parse_args()
    return args
def generate_train_dataset(n_samples = 100_000,chain_type='heavy'):
    if chain_type == 'heavy':
        heavy_augmentor = HeavyChainSequenceAugmentor(data_config_builtin, SequenceAugmentorArguments())
    elif chain_type == 'light':
        heavy_augmentor = LightChainKappaLambdaSequenceAugmentor(kappa_dataconfig=kappa_data_config_builtin,
                                                           lambda_dataconfig=lambda_data_config_builtin,
                                                           lambda_args=SequenceAugmentorArguments(),
                                                           kappa_args=SequenceAugmentorArguments())



    train_sequences = []
    train_labels = []

    logging.info('Staring To Generate S5F Portion of Train Dataset')
    for _ in tqdm(range(n_samples//2)):
        heavy_sequence = heavy_augmentor.simulate_augmented_sequence()
        label = random.choice(["Normal", 'Reversed', 'Complement', 'Reverse Complement'])
        train_labels.append(label)
        if label == 'Normal':
            train_sequences.append(heavy_sequence['sequence'])
        elif label == 'Reversed':
            train_sequences.append(reverse_sequence(heavy_sequence['sequence']))
        elif label == 'Complement':
            train_sequences.append(complement_sequence(heavy_sequence['sequence']))
        elif label == 'Reverse Complement':
            train_sequences.append(reverse_complement_sequence(heavy_sequence['sequence']))
    heavy_augmentor = HeavyChainSequenceAugmentor(data_config_builtin,
                                                  SequenceAugmentorArguments(mutation_model=Uniform))

    logging.info('Staring To Generate Uniform Portion of Train Dataset')
    for _ in tqdm(range(n_samples//2)):
        heavy_sequence = heavy_augmentor.simulate_augmented_sequence()
        label = random.choice(["Normal", 'Reversed', 'Complement', 'Reverse Complement'])
        train_labels.append(label)
        if label == 'Normal':
            train_sequences.append(heavy_sequence['sequence'])
        elif label == 'Reversed':
            train_sequences.append(reverse_sequence(heavy_sequence['sequence']))
        elif label == 'Complement':
            train_sequences.append(complement_sequence(heavy_sequence['sequence']))
        elif label == 'Reverse Complement':
            train_sequences.append(reverse_complement_sequence(heavy_sequence['sequence']))

    logging.info('Finished Generating Train Dataset!')

    return train_sequences,train_labels

def generate_uniform_test_dataset(n_samples = 100_000,chain_type='heavy'):
    if chain_type == 'heavy':
        heavy_augmentor = HeavyChainSequenceAugmentor(data_config_builtin, SequenceAugmentorArguments(mutation_model=Uniform))
    elif chain_type == 'light':
        heavy_augmentor = LightChainKappaLambdaSequenceAugmentor(kappa_dataconfig=kappa_data_config_builtin,
                                                                 lambda_dataconfig=lambda_data_config_builtin,
                                                                 lambda_args=SequenceAugmentorArguments(mutation_model=Uniform),
                                                                 kappa_args=SequenceAugmentorArguments(mutation_model=Uniform))
    uniform_test_sequences = []
    uniform_test_labels = []

    logging.info('Starting to Generate Uniform Mutation Model Test Samples')
    for _ in tqdm(range(n_samples)):
        heavy_sequence = heavy_augmentor.simulate_augmented_sequence()
        label = random.choice(["Normal", 'Reversed', 'Complement', 'Reverse Complement'])
        uniform_test_labels.append(label)
        if label == 'Normal':
            uniform_test_sequences.append(heavy_sequence['sequence'])
        elif label == 'Reversed':
            uniform_test_sequences.append(reverse_sequence(heavy_sequence['sequence']))
        elif label == 'Complement':
            uniform_test_sequences.append(complement_sequence(heavy_sequence['sequence']))
        elif label == 'Reverse Complement':
            uniform_test_sequences.append(reverse_complement_sequence(heavy_sequence['sequence']))

    logging.info('Finished to Generate Uniform Mutation Model Test Samples')

    return uniform_test_sequences,uniform_test_labels

def generate_partial_uniform_test_dataset(n_samples = 100_000,chain_type='heavy'):
    if chain_type == 'heavy':
        heavy_augmentor = HeavyChainSequenceAugmentor(data_config_builtin, SequenceAugmentorArguments(mutation_model=Uniform))
    elif chain_type == 'light':
        heavy_augmentor = LightChainKappaLambdaSequenceAugmentor(kappa_dataconfig=kappa_data_config_builtin,
                                                                 lambda_dataconfig=lambda_data_config_builtin,
                                                                 lambda_args=SequenceAugmentorArguments(mutation_model=Uniform),
                                                                 kappa_args=SequenceAugmentorArguments(mutation_model=Uniform))

    partial_uniform_test_sequences = []
    partial_uniform_test_labels = []
    partial_labels = []
    sampling_type_set = ['Only V', 'Only D', 'Only J', 'VD', 'VJ', 'DJ']
    if chain_type == 'light':
        sampling_type_set = ['Only V', 'Only J', 'VJ']

    def get_partial_sequence(simulated,sampling_type_set):
        partial_label = random.choice(sampling_type_set)
        sequence = None
        if partial_label == 'Only V':
            sequence = simulated['sequence'][simulated['v_sequence_start']:simulated['v_sequence_end']]
        elif partial_label == 'Only D':
            sequence = simulated['sequence'][simulated['d_sequence_start']:simulated['d_sequence_end']]
        elif partial_label == 'Only J':
            sequence = simulated['sequence'][simulated['j_sequence_start']:simulated['j_sequence_end']]
        elif partial_label == 'VD':
            sequence = simulated['sequence'][simulated['v_sequence_start']:simulated['v_sequence_end']] + \
                       simulated['sequence'][simulated['d_sequence_start']:simulated['d_sequence_end']]
        elif partial_label == 'VJ':
            sequence = simulated['sequence'][simulated['v_sequence_start']:simulated['v_sequence_end']] + \
                       simulated['sequence'][simulated['j_sequence_start']:simulated['j_sequence_end']]
        elif partial_label == 'DJ':
            sequence = simulated['sequence'][simulated['d_sequence_start']:simulated['d_sequence_end']] + \
                       simulated['sequence'][simulated['j_sequence_start']:simulated['j_sequence_end']]

        return sequence, partial_label

    logging.info('Starting to Generate Uniform Mutation Model Partial Test Samples')

    for _ in tqdm(range(n_samples)):
        heavy_sequence = heavy_augmentor.simulate_augmented_sequence()
        sequence, partial_label = get_partial_sequence(heavy_sequence,sampling_type_set)
        partial_labels.append(partial_label)
        heavy_sequence['sequence'] = sequence
        label = random.choice(["Normal", 'Reversed', 'Complement', 'Reverse Complement'])
        partial_uniform_test_labels.append(label)
        if label == 'Normal':
            partial_uniform_test_sequences.append(heavy_sequence['sequence'])
        elif label == 'Reversed':
            partial_uniform_test_sequences.append(reverse_sequence(heavy_sequence['sequence']))
        elif label == 'Complement':
            partial_uniform_test_sequences.append(complement_sequence(heavy_sequence['sequence']))
        elif label == 'Reverse Complement':
            partial_uniform_test_sequences.append(reverse_complement_sequence(heavy_sequence['sequence']))

    logging.info('Finished to Generate Uniform Mutation Model Partial Test Samples')

    return  partial_uniform_test_sequences,partial_uniform_test_labels,partial_labels
def create_and_fit_pipeline(train_sequences,train_labels,min_k=3,max_k=5,max_features=576):

    logging.info('Creating Prediction Pipeline')

    orient_pipeline = Pipeline(
        [('vectorizer', CountVectorizer(analyzer='char',
                                        ngram_range=(min_k, max_k),
                                        max_features=max_features,
                                        binary=True)),
         ('model', LogisticRegression(random_state=42))])

    logging.info('Fitting Prediction Pipeline')
    orient_pipeline.fit(train_sequences, train_labels)
    logging.info('Pipeline was Fitted Successfully!')
    return orient_pipeline
def evaluate_model(orient_pipeline,test_sequences,test_labels,title,fname,save_path_figure):
    logging.info('Evaluating Model on Test Dataset')
    predictions = orient_pipeline.predict(test_sequences)
    logging.info('Predictions Produced...Creating and Saving Confusion Matrix')
    plot_confusion_matrix(test_labels, predictions, x_tick_rotation=90, normalize=True)
    plt.title(title)
    plt.savefig(save_path_figure+fname, bbox_inches='tight', dpi=150)
    logging.info(f'Confusion Matrix was Saved at: {save_path_figure+fname}')
    return predictions
def save_pipeline(pipeline,save_path):
    logging.info('Saving Pipeline...')
    with open(save_path+'Custom_DNA_Orientation_Pipeline.pkl','wb') as h:
        pickle.dump(pipeline,h)
    logging.info(f"Pipeline was Saved at: {save_path+'Custom_DNA_Orientation_Pipeline.pkl'}")

def main():
    args = parse_arguments()

    train_sequences,train_labels = generate_train_dataset(n_samples=args.train_dataset_size,chain_type=args.chain_type)
    uniform_test_sequences, uniform_test_labels = generate_uniform_test_dataset(n_samples=args.uniform_test_dataset_size,
                                                                                chain_type=args.chain_type)
    partial_uniform_test_sequences, partial_uniform_test_labels, partial_labels = \
        generate_partial_uniform_test_dataset(n_samples=args.partial_test_dataset_size,chain_type=args.chain_type)

    pipeline = create_and_fit_pipeline(train_sequences,train_labels)
    evaluate_model(pipeline,uniform_test_sequences,uniform_test_labels,'Uniform Mutation Model Sequences',
                   'uniform_mm_normalized_confusion_matrix.png',args.plot_save_path)
    evaluate_model(pipeline, partial_uniform_test_sequences, partial_uniform_test_labels,
                   'Uniform Mutation Model Prtial Sequences',
                   'partial_uniform_mm_normalized_confusion_matrix.png', args.plot_save_path)

    save_pipeline(pipeline,args.save_path)

if __name__ == '__main__':
    main()

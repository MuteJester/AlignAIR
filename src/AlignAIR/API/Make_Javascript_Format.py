import logging

import numpy as np
import argparse
import pickle

from prompt_toolkit.filters import is_multiline

from AlignAIR.Data import MultiDataConfigContainer, MultiChainDataset, SingleChainDataset
from AlignAIR.Data.PredictionDataset import PredictionDataset
from AlignAIR.Models import MultiChainAlignAIR, SingleChainAlignAIR
from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Preprocessing.Steps.dataconfig_steps import ConfigLoadStep
from AlignAIR.Preprocessing.Steps.file_steps import FileSampleCounterStep, FileNameExtractionStep
from AlignAIR.Preprocessing.Steps.model_loading_steps import ModelLoadingStep
from AlignAIR.Step.Step import Step
from AlignAIR.Trainers import Trainer


def parse_arguments():
    """
        Parse command line arguments.

        Returns:
            argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='AlignAIR Model Save Type Conversion ')
    parser.add_argument('--genairr_dataconfig', type=str, default='HUMAN_IGH_OGRDB', help='A name of a builtin GenAIRR data config, or a path to a custom data config pkl file, in the case of a multi chain model this should be a comma separated list of configs, e.g. "HUMAN_IGH_OGRDB,HUMAN_TCRB_IMGT", can also be paths to custom data config pkl files')
    parser.add_argument('--model_checkpoint', type=str, help='Path to saved AlignAIR weights',required=True)
    parser.add_argument('--save_path', type=str, help='Where to save the new foramt',required=True)
    parser.add_argument('--sequences', type=str, help='Path to csv/tsv/fasta file with sequences in a column called "sequence"',required=True)
    parser.add_argument('--max_input_size', type=int, default=576, help='Maximum model input size, NOTE! this is with respect to the dimensions the model was trained on, do not increase for pretrained models')
    parser.add_argument('--batch_size', type=int, default=2048, help='The Batch Size for The Model Prediction')
    parser.add_argument('--translate_to_asc', action='store_true', help='Translate names back to ASCs names from IMGT')
    parser.add_argument('--fix_orientation', type=bool, default=True, help='Adds a preprocessing steps that tests and fixes the DNA orientation, in case it is reversed, complement or reversed and complement')
    parser.add_argument('--custom_orientation_pipeline_path', type=str, default=None, help='A path to a custom orientation model created for a custom reference')
    parser.add_argument('--custom_genotype', type=str, default=None, help='Path to a custom genotype yaml file')
    parser.add_argument('--save_predict_object', action='store_true', help='Save the predict object (Warning this can be large)')
    parser.add_argument('--airr_format', action='store_true', help='Adds a step to format the results to AIRR format')
    # Note: finetuned_model_params_yaml is deprecated; fine-tune and save a new bundle instead.

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('PipelineLogger')
    # mount logger to all step objects
    Step.set_logger(logger)
    # Parse command line arguments
    args = parse_arguments()
    # Process arguments based on mode
    config = args

    predict_object = PredictObject(config, logger=logger)

    steps = [
        ConfigLoadStep("Load Config"),
        FileNameExtractionStep('Get File Name'),
        FileSampleCounterStep('Count Samples in File'),
        ModelLoadingStep('Load Models'),
    ]

    for step in steps:
        predict_object = step.execute(predict_object)

    model = predict_object.model
    if model is None:
        raise RuntimeError("Model was not loaded. Ensure ModelLoadingStep completed successfully and arguments are valid.")

    prediction_Dataset = PredictionDataset(max_sequence_length=args.max_input_size)
    seq = 'CAGCCACAACTGAACTGGTCAAGTCCAGGACTGGTGAATACCTCGCAGACCGTCACACTCACCCTTGCCGTGTCCGGGGACCGTGTCTCCAGAACCACTGCTGTTTGGAAGTGGAGGGGTCAGACCCCATCGCGAGGCCTTGCGTGGCTGGGAAGGACCTACNACAGTTCCAGGTGATTTGCTAACAACGAAGTGTCTGTGAATTGTTNAATATCCATGAACCCAGACGCATCCANGGAACGGNTCTTCCTGCACCTGAGGTCTGGGGCCTTCGACGACACGGCTGTACATNCGTGAGAAAGCGGTGACCTCTACTAGGATAGTGCTGAGTACGACTGGCATTACGCTCTCNGGGACCGTGCCACCCTTNTCACTGCCTCCTCGG'
    es = prediction_Dataset.encode_and_equal_pad_sequence(seq)['tokenized_sequence']
    predicted = model.predict({'tokenized_sequence': np.vstack([es])})

    dummy_input = {
        "tokenized_sequence": np.zeros((1, args.max_input_size), dtype=np.float32),
    }
    _ = model(dummy_input)  # Build the model by invoking it

    model.save(args.save_path)
    # validate folder in saved path
    import os
    status = os.path.exists(args.save_path)
    print("Converted model saved in folder: ", status)
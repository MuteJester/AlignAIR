import argparse
import logging
import yaml
import questionary
import tensorflow as tf
from AlignAIR.PostProcessing.Steps.allele_threshold_step import MaxLikelihoodPercentageThresholdApplicationStep, \
    ConfidenceMethodThresholdApplicationStep
from AlignAIR.PostProcessing.Steps.clean_up_steps import CleanAndArrangeStep
from AlignAIR.PostProcessing.Steps.finalization_and_packaging_steps import FinalizationStep
from AlignAIR.PostProcessing.Steps.germline_alignment_steps import AlleleAlignmentStep
from AlignAIR.PostProcessing.Steps.segmentation_correction_steps import SegmentCorrectionStep
from AlignAIR.PostProcessing.Steps.translate_to_imgt_step import TranslationStep
from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Preprocessing.Steps.batch_processing_steps import BatchProcessingStep
from AlignAIR.Preprocessing.Steps.dataconfig_steps import ConfigLoadStep
from AlignAIR.Preprocessing.Steps.file_steps import FileNameExtractionStep, FileSampleCounterStep
from AlignAIR.Preprocessing.Steps.model_loading_steps import ModelLoadingStep

# Set TensorFlow logging level to ERROR
tf.get_logger().setLevel('ERROR')
class Args:
    """
       A class to convert dictionary entries to class attributes.
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)

def parse_arguments():
    """
        Parse command line arguments.

        Returns:
            argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='AlignAIR Model Prediction')
    parser.add_argument('--mode', type=str, default='cli', choices=['cli', 'yaml', 'interactive'],
                        help='Mode of input: cli, yaml, interactive')
    parser.add_argument('--config_file', type=str, help='Path to YAML configuration file')
    parser.add_argument('--model_checkpoint', type=str, help='Path to saved AlignAIR weights',required=True)
    parser.add_argument('--save_path', type=str, help='Where to save the alignment',required=True)
    parser.add_argument('--chain_type', type=str, help='heavy / light',required=True)
    parser.add_argument('--sequences', type=str, help='Path to csv/tsv/fasta file with sequences in a column called "sequence"',required=True)
    parser.add_argument('--lambda_data_config', type=str, default='D', help='Path to lambda chain data config')
    parser.add_argument('--kappa_data_config', type=str, default='D', help='Path to kappa chain data config')
    parser.add_argument('--heavy_data_config', type=str, default='D', help='Path to heavy chain data config')
    parser.add_argument('--max_input_size', type=int, default=576, help='Maximum model input size, NOTE! this is with respect to the dimensions the model was trained on, do not increase for pretrained models')
    parser.add_argument('--batch_size', type=int, default=2048, help='The Batch Size for The Model Prediction')
    parser.add_argument('--v_allele_threshold', type=float, default=0.1, help='Percentage for V allele assignment '
                                                                              'selection')
    parser.add_argument('--d_allele_threshold', type=float, default=0.1, help='Percentage for D allele assignment selection')
    parser.add_argument('--j_allele_threshold', type=float, default=0.1, help='Percentage for J allele assignment selection')
    parser.add_argument('--v_cap', type=int, default=3, help='Cap for V allele calls')
    parser.add_argument('--d_cap', type=int, default=3, help='Cap for D allele calls')
    parser.add_argument('--j_cap', type=int, default=3, help='Cap for J allele calls')
    parser.add_argument('--translate_to_asc', action='store_true', help='Translate names back to ASCs names from IMGT')
    parser.add_argument('--fix_orientation', type=bool, default=True, help='Adds a preprocessing steps that tests and fixes the DNA orientation, in case it is reversed, complement or reversed and complement')
    parser.add_argument('--custom_orientation_pipeline_path', type=str, default=None, help='A path to a custom orientation model created for a custom reference')

    return parser.parse_args()

def load_yaml_config(config_file):
    """
        Load configuration from a YAML file.

        Args:
            config_file (str): Path to the YAML configuration file.

        Returns:
            Args: Configuration loaded into an Args object.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return Args(**config)

def interactive_mode():
    """
        Collect configuration interactively using questionary.

        Returns:
            Args: Configuration collected interactively.
    """
    config = {}
    config['model_checkpoint'] = questionary.text("Path to saved AlignAIR weights:").ask()
    config['save_path'] = questionary.text("Where to save the alignment:").ask()
    config['chain_type'] = questionary.select("Chain type:", choices=['heavy', 'light']).ask()
    config['sequences'] = questionary.text("Path to csv/tsv file with sequences in a column called 'sequence':").ask()
    config['lambda_data_config'] = questionary.text("Path to lambda chain data config:", default='D').ask()
    config['kappa_data_config'] = questionary.text("Path to kappa chain data config:", default='D').ask()
    config['heavy_data_config'] = questionary.text("Path to heavy chain data config:", default='D').ask()
    config['max_input_size'] = int(questionary.text("Maximum model input size:", default='576').ask())
    config['batch_size'] = int(questionary.text("Batch size for the model prediction:", default='2048').ask())
    config['v_allele_threshold'] = float(questionary.text("Threshold for V allele prediction:", default='0.75').ask())
    config['d_allele_threshold'] = float(questionary.text("Threshold for D allele prediction:", default='0.3').ask())
    config['j_allele_threshold'] = float(questionary.text("Threshold for J allele prediction:", default='0.8').ask())
    config['v_cap'] = int(questionary.text("Cap for V allele calls:", default='3').ask())
    config['d_cap'] = int(questionary.text("Cap for D allele calls:", default='3').ask())
    config['j_cap'] = int(questionary.text("Cap for J allele calls:", default='3').ask())
    config['translate_to_asc'] = questionary.confirm("Translate names back to ASCs names from IMGT?").ask()
    config['fix_orientation'] = questionary.confirm("Fix DNA orientation if reversed or complement?").ask()
    config['custom_orientation_pipeline_path'] = questionary.text("Path to a custom orientation model:", default='').ask()
    return Args(**config)

def run_pipeline(predict_object, steps):
    """
        Execute a series of processing steps on the predict object.

        Args:
            predict_object (PredictObject): The object to be processed.
            steps (list): List of processing steps to execute.
    """
    for step in steps:
        predict_object = step.execute(predict_object)

def main():
    """
        Main function to execute the AlignAIR prediction pipeline.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('PipelineLogger')
    args = parse_arguments()

    if args.mode == 'cli':
        config = args
    elif args.mode == 'yaml':
        if not args.config_file:
            raise ValueError("YAML mode requires --config_file argument")
        config = load_yaml_config(args.config_file)
    elif args.mode == 'interactive':
        config = interactive_mode()

    predict_object = PredictObject(config, logger=logger)

    steps = [
        ConfigLoadStep("Load Config", logger),
        FileNameExtractionStep('Get File Name', logger),
        FileSampleCounterStep('Count Samples in File', logger),
        ModelLoadingStep('Load Models', logger),
        BatchProcessingStep("Process and Predict Batches", logger),
        CleanAndArrangeStep("Clean Up Raw Prediction", logger),
        SegmentCorrectionStep("Correct Segmentations", logger),
        MaxLikelihoodPercentageThresholdApplicationStep("Apply Max Likelihood Threshold to Distill Assignments", logger),
        AlleleAlignmentStep("Align Predicted Segments with Germline", logger),
        TranslationStep("Translate ASC's to IMGT Alleles", logger),
        FinalizationStep("Finalize Post Processing and Save Csv", logger)
    ]

    run_pipeline(predict_object, steps)
    logger.info("Pipeline execution complete.")

if __name__ == '__main__':
    main()

import argparse
import logging
import yaml
import os
import sys
import tensorflow as tf
from AlignAIR.PostProcessing.Steps.allele_threshold_step import MaxLikelihoodPercentageThresholdApplicationStep, \
    ConfidenceMethodThresholdApplicationStep
from AlignAIR.PostProcessing.Steps.clean_up_steps import CleanAndArrangeStep
from AlignAIR.PostProcessing.Steps.correct_likelihood_for_genotype_step import GenotypeBasedLikelihoodAdjustmentStep
from AlignAIR.PostProcessing.Steps.finalization_and_packaging_steps import FinalizationStep
from AlignAIR.PostProcessing.Steps.airr_finalization_and_packaging_steps import AIRRFinalizationStep
from AlignAIR.PostProcessing.Steps.germline_alignment_steps import AlleleAlignmentStep
from AlignAIR.PostProcessing.Steps.segmentation_correction_steps import SegmentCorrectionStep
from AlignAIR.PostProcessing.Steps.translate_to_imgt_step import TranslationStep
from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Preprocessing.Steps.batch_processing_steps import BatchProcessingStep
from AlignAIR.Preprocessing.Steps.dataconfig_steps import ConfigLoadStep
from AlignAIR.Preprocessing.Steps.file_steps import FileNameExtractionStep, FileSampleCounterStep
from AlignAIR.Preprocessing.Steps.model_loading_steps import ModelLoadingStep
from AlignAIR.Step.Step import Step

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
        Parse command line arguments for CLI mode.

        Returns:
            argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='AlignAIR Model Prediction')
    # Bundle is the single source of truth (contains weights, config, dataconfig, input size)
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to a pretrained model bundle directory (contains config.json, weights, dataconfig).')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory where the alignment outputs will be saved.')
    parser.add_argument('--sequences', type=str, required=True,
                        help='Path to csv/tsv/fasta file with sequences in a column called "sequence"')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for prediction (default: 2048).')
    parser.add_argument('--v_allele_threshold', type=float, default=0.1, help='Percentage for V allele assignment (default: 0.1).')
    parser.add_argument('--d_allele_threshold', type=float, default=0.1, help='Percentage for D allele assignment (default: 0.1).')
    parser.add_argument('--j_allele_threshold', type=float, default=0.1, help='Percentage for J allele assignment (default: 0.1).')
    parser.add_argument('--v_cap', type=int, default=3, help='Cap for V allele calls (default: 3).')
    parser.add_argument('--d_cap', type=int, default=3, help='Cap for D allele calls (default: 3).')
    parser.add_argument('--j_cap', type=int, default=3, help='Cap for J allele calls (default: 3).')
    parser.add_argument('--translate_to_asc', action='store_true', help='Translate names back to ASCs names from IMGT.')
    # Safer boolean pair for fix_orientation (default True), allow disabling with --no_fix_orientation
    parser.add_argument('--fix_orientation', dest='fix_orientation', action='store_true', default=True,
                        help='Enable orientation check and fix (default: enabled).')
    parser.add_argument('--no_fix_orientation', dest='fix_orientation', action='store_false',
                        help='Disable orientation check and fix.')
    parser.add_argument('--custom_orientation_pipeline_path', type=str, default=None,
                        help='Path to a custom orientation model created for a custom reference.')
    parser.add_argument('--custom_genotype', type=str, default=None, help='Path to a custom genotype YAML file.')
    parser.add_argument('--save_predict_object', action='store_true', help='Save the predict object (warning: can be large).')
    parser.add_argument('--airr_format', action='store_true', help='Format results to AIRR standard.')
    # NOTE: fine-tune head sizes via training and save a new bundle. Deprecated: --finetuned_model_params_yaml

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
    # mount logger to all step objects
    Step.set_logger(logger)

    try:
        # --- Automatic mode detection: YAML file as sole argument ---
        if len(sys.argv) == 2 and (sys.argv[1].endswith('.yaml') or sys.argv[1].endswith('.yml')):
            config_path = sys.argv[1]
            logger.info(f"YAML configuration detected: {config_path}")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            config = load_yaml_config(config_path)
        else:
            # CLI mode
            args = parse_arguments()
            config = args

        # --- Basic validation and setup ---
        model_dir = getattr(config, 'model_dir', None)
        sequences_path = getattr(config, 'sequences', None)
        save_path = getattr(config, 'save_path', None)

        if not model_dir or not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model bundle directory not found or not specified: {model_dir}")
        if not sequences_path or not os.path.exists(sequences_path):
            raise FileNotFoundError(f"Sequences file not found or not specified: {sequences_path}")
        if not save_path:
            raise ValueError("--save_path must be provided")
        os.makedirs(save_path, exist_ok=True)

        predict_object = PredictObject(config, logger=logger)

        steps = [
            ConfigLoadStep("Load Config"),
            FileNameExtractionStep('Get File Name'),
            FileSampleCounterStep('Count Samples in File'),
            ModelLoadingStep('Load Models'),
            BatchProcessingStep("Process and Predict Batches"),
            CleanAndArrangeStep("Clean Up Raw Prediction"),
            GenotypeBasedLikelihoodAdjustmentStep("Adjust Likelihoods for Genotype"),
            SegmentCorrectionStep("Correct Segmentations"),
            MaxLikelihoodPercentageThresholdApplicationStep("Apply Max Likelihood Threshold to Distill Assignments"),
            AlleleAlignmentStep("Align Predicted Segments with Germline")
        ]

        if getattr(config, 'airr_format', False):
            steps.append(AIRRFinalizationStep("Finalize Results"))
        else:
            steps.append(TranslationStep("Translate ASC's to IMGT Alleles"))
            steps.append(FinalizationStep("Finalize Results"))

        run_pipeline(predict_object, steps)
        logger.info("Pipeline execution complete.")

        if getattr(config, 'save_predict_object', False):
            save_path = predict_object.script_arguments.save_path
            file_name = predict_object.file_info.file_name
            path = os.path.join(save_path, f"{file_name}_alignair_results_predictObject.pkl")
            logger.info('Detaching model from predict object before saving')
            predict_object.model = None
            predict_object.save(path)
            logger.info("Predict Object Saved At: {}".format(path))

    except Exception as e:
        logger.error(f"Error during prediction pipeline: {e}")
        raise


if __name__ == '__main__':
    main()

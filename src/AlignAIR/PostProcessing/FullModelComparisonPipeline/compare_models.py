import argparse
import logging
import multiprocessing
import tensorflow as tf
from AlignAIR.PostProcessing.FullModelEvaluationPipeline.create_likelihood_figure_step import \
    ModelLikelihoodSummaryPlotStep
from AlignAIR.PostProcessing.FullModelEvaluationPipeline.create_mutation_rate_figure_step import \
    MutationRateSummaryPlotStep
from AlignAIR.PostProcessing.FullModelEvaluationPipeline.create_segmentation_and_productivity_figure_step import \
    SegmentationProductivitySummaryPlotStep
from AlignAIR.PostProcessing.OptimalAlleleThresholdSearch.find_threshold_step import OptimalAlleleThresholdSearchStep
from AlignAIR.PostProcessing.Steps.allele_threshold_step import ThresholdApplicationStep
from AlignAIR.PostProcessing.Steps.clean_up_steps import CleanAndArrangeStep
from AlignAIR.PostProcessing.Steps.finalization_and_packaging_steps import FinalizationStep
from AlignAIR.PostProcessing.Steps.germline_alignment_steps import AlleleAlignmentStep
from AlignAIR.PostProcessing.Steps.segmentation_correction_steps import SegmentCorrectionStep
from AlignAIR.PostProcessing.Steps.translate_to_imgt_step import TranslationStep
from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Preprocessing.Steps.batch_processing_steps import BatchProcessingStep
from AlignAIR.Preprocessing.Steps.dataconfig_steps import ConfigLoadStep
from AlignAIR.Preprocessing.Steps.file_steps import FileNameExtractionStep, FileSampleCounterStep
from AlignAIR.Preprocessing.Steps.load_groundtruth_table_step import LoadGroundTruthStep
from AlignAIR.Preprocessing.Steps.model_loading_steps import ModelLoadingStep
import platform

tf.get_logger().setLevel('ERROR')


# Setup logging

def parse_arguments():
    parser = argparse.ArgumentParser(description='AlingAIR Model Prediction')
    parser.add_argument('--model_checkpoint_a', type=str, required=True, help='path to saved alignair weights')
    parser.add_argument('--model_checkpoint_b', type=str, required=True, help='path to saved alignair weights')
    parser.add_argument('--model_checkpoint', type=str, help='path to saved alignair weights')
    parser.add_argument('--save_path', type=str, required=True, help='where to save the results')
    parser.add_argument('--chain_type', type=str, required=True, help='heavy / light')
    parser.add_argument('--sequences', type=str, required=True,
                        help='path to csv/tsv file with sequences and ground truth data ')

    parser.add_argument('--lambda_data_config', type=str, default='D', help='path to lambda chain data config')
    parser.add_argument('--kappa_data_config', type=str, default='D', help='path to  kappa chain data config')
    parser.add_argument('--heavy_data_config', type=str, default='D', help='path to heavy chain  data config')
    parser.add_argument('--max_input_size', type=int, default=576,
                        help='maximum model input size, NOTE! this is with respect to the dimensions the model was trained on, do not increase for pretrained models')
    parser.add_argument('--batch_size', type=int, default=2048, help='The Batch Size for The Model Prediction')

    # For Pre Processing
    parser.add_argument('--fix_orientation', type=bool, default=True,
                        help='Adds a preprocessing steps that tests and fixes the DNA orientation, in case it is '
                             'reversed,compliment or reversed and compliment')
    parser.add_argument('--custom_orientation_pipeline_path', type=str, default=None,
                        help='a path to a custom orientation model created for a custom reference')

    args = parser.parse_args()
    return args


def run_pipeline(predict_object, steps):
    for step in steps:
        predict_object = step.execute(predict_object)


def main():
    # Setup initial PredictObject
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('PipelineLogger')
    args = parse_arguments()

    args_a = args

    predict_object_a = PredictObject(args, logger=logger)
    predict_object_b = PredictObject(args, logger=logger)

    # Define the steps in the pipeline
    steps = [
        ConfigLoadStep("Load Config", logger),
        FileNameExtractionStep('Get File Name', logger),
        FileSampleCounterStep('Count Samples in File', logger),
        LoadGroundTruthStep('Load Ground Truth Table', logger),
        ModelLoadingStep('Load Models', logger),
        BatchProcessingStep("Process and Predict Batches", logger),  # Predict via model
        CleanAndArrangeStep("Clean Up Raw Prediction", logger),
        SegmentCorrectionStep("Correct Segmentations", logger),
        OptimalAlleleThresholdSearchStep("Optimal Allele Search", logger),
        ThresholdApplicationStep("Apply Dynamic Threshold to Distill Assignments", logger),
        ModelLikelihoodSummaryPlotStep('Generate Model Likelihood Function Figure', logger),
        SegmentationProductivitySummaryPlotStep('Generate Model Segmentation&Productivity Function Figure', logger),
        MutationRateSummaryPlotStep('Generate Model Mutation Rate Figure', logger),
    ]

    # Run the pipeline
    final_predict_object_a = run_pipeline(predict_object_a, steps)
    logger.info("Pipeline execution complete for model A.")
    logger.info("Starting Pipeline for model B")

    final_predict_object_b = run_pipeline(predict_object_b, steps)




if __name__ == '__main__':
    main()

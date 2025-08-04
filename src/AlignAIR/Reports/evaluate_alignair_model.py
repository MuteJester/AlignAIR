
import argparse
import pickle
import pandas as pd
import os
from .utils.file_management import load_pickle, load_dataconfig
from .plots.likelihoods import LikelihoodCalibrationPlot
from .plots.performance_metrics import PerformanceMetricsPlot, SequenceAnalysisPlot, AlleleFrequencyPlot
from .plots.confidence_analysis import ConfidenceAnalysisPlot, ThresholdEffectsPlot
from .plots.error_analysis import ErrorAnalysisPlot
from .plots.report_generator import AlignAIRReportGenerator

def parse_args():
    """Parses command-line arguments for evaluating an AlignAIRR model."""
    parser = argparse.ArgumentParser(
        description="Evaluate an AlignAIRR model on a test dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ground_truth_table", required=True, help="Path to the test dataset file.")
    parser.add_argument("--genairr_dataconfig", required=True, help="GenAIRR dataconfig for the test dataset.")
    parser.add_argument("--predict_object_path", required=True, help="Path to the prediction object pickle file.")
    # model path is optional, allowing for evaluation of existing models
    parser.add_argument("--model_path", default=None, help="Path to the AlignAIRR model to evaluate. If not provided, the script will look for a default model in the session path.")
    # max sequence length is optional, defaulting to 576
    parser.add_argument("--max_sequence_length", type=int, default=576,
                        help="Maximum input sequence length for the model. Default is 576.")
    # save path for the HTML report
    parser.add_argument("--save_path", required=True, help="Directory path where the evaluation report will be saved.")
    # optional custom report filename
    parser.add_argument("--report_filename", default=None, help="Custom filename for the report. If not provided, a timestamp-based name will be used.")

    return parser.parse_args()


def create_comprehensive_report(predict_object, ground_truth_table, genairr_dataconfig, save_path, report_filename=None):
    """Create a comprehensive evaluation report with multiple plot types"""
    
    # Initialize the report generator
    print("Initializing AlignAIR evaluation report...")
    report_generator = AlignAIRReportGenerator(predict_object, ground_truth_table, genairr_dataconfig)
    
    # Add different types of analysis plots
    print("Generating analysis plots...")
    
    # 1. Likelihood Calibration Analysis
    report_generator.add_plot(
        LikelihoodCalibrationPlot,
        title="Likelihood Calibration Analysis",
        description="Analyzes how well the predicted likelihoods are calibrated against actual performance. "
                   "Points closer to the diagonal line indicate better calibration. Includes confidence intervals "
                   "and performance vs mutation rate analysis for V, D, and J genes."
    )
    
    # 2. Performance Metrics Analysis
    report_generator.add_plot(
        PerformanceMetricsPlot,
        title="Performance Metrics Overview",
        description="Comprehensive performance analysis including gene-wise accuracy comparison, "
                   "Jaccard similarity distributions, and individual gene performance trends. "
                   "Provides both summary statistics and detailed per-gene analysis."
    )
    
    # 3. Sequence Characteristics Analysis
    report_generator.add_plot(
        SequenceAnalysisPlot,
        title="Sequence Characteristics Analysis",
        description="Analyzes the relationship between sequence characteristics (length, mutation rate) "
                   "and prediction performance. Helps identify if certain sequence properties affect "
                   "model accuracy and provides insights into model behavior across different sequence types."
    )
    
    # 4. Allele Frequency Analysis
    report_generator.add_plot(
        AlleleFrequencyPlot,
        title="Allele Frequency Distribution",
        description="Compares the frequency distribution of alleles in ground truth vs predictions. "
                   "Helps identify if the model has bias towards certain alleles and whether the "
                   "prediction distribution matches the true biological distribution."
    )
    
    # 5. Confidence Analysis
    report_generator.add_plot(
        ConfidenceAnalysisPlot,
        title="Confidence and Uncertainty Analysis",
        description="Analyzes the confidence scores and uncertainty measures of predictions. "
                   "Includes maximum likelihood distributions, entropy analysis, and the relationship "
                   "between confidence and uncertainty. Higher entropy indicates more uncertain predictions."
    )
    
    # 6. Threshold Effects Analysis
    report_generator.add_plot(
        ThresholdEffectsPlot,
        title="Threshold Effects on Performance",
        description="Analyzes how different threshold settings affect prediction performance. "
                   "Shows the trade-off between accuracy and number of calls per sequence, "
                   "helping to optimize threshold selection for different use cases."
    )
    
    # 7. Deep Learning Model Error Analysis
    success = report_generator.add_plot(
        ErrorAnalysisPlot,
        title="Deep Learning Model Error Analysis",
        description="Comprehensive analysis of prediction errors across the entire AlignAIR pipeline. "
                   "Examines segmentation boundary errors, allele assignment mistakes, indel regression analysis, "
                   "mutation rate prediction accuracy, productivity classification errors, and sequence length impacts. "
                   "Provides insights into model weaknesses and areas for improvement using actual model outputs."
    )
    
    # Generate and save the report
    print("Compiling HTML report...")
    
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Save the report
    saved_path = report_generator.save_report(save_path, report_filename)
    
    if saved_path:
        print(f"✓ Evaluation report saved: {saved_path}")
        return saved_path
    else:
        print("✗ Failed to save evaluation report")
        return None


if __name__ == "__main__":
    args = parse_args()

    print("Loading evaluation data...")
    predict_object = load_pickle(args.predict_object_path)
    ground_truth_table = pd.read_csv(args.ground_truth_table)
    genairr_dataconfig = load_dataconfig(args.genairr_dataconfig)

    print(f"Loaded {len(ground_truth_table):,} sequences for evaluation")
    
    # Create comprehensive report
    report_path = create_comprehensive_report(
        predict_object=predict_object,
        ground_truth_table=ground_truth_table,
        genairr_dataconfig=genairr_dataconfig,
        save_path=args.save_path,
        report_filename=args.report_filename
    )
    
    if report_path:
        print(f"Report saved successfully to: {report_path}")
    else:
        print("Report generation failed")
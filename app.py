import argparse
import logging
import os
import shlex
import sys
import yaml
import questionary
import tensorflow as tf

from AlignAIR.PostProcessing.Steps.allele_threshold_step import MaxLikelihoodPercentageThresholdApplicationStep, \
    ConfidenceMethodThresholdApplicationStep
from AlignAIR.PostProcessing.Steps.clean_up_steps import CleanAndArrangeStep
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
import psutil
import platform
import tensorflow as tf




# ANSI color codes for styling
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

tf.get_logger().setLevel('ERROR')


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def display_header():
    """Display the header with a B cell receptor ASCII art and welcome message."""
    art = f"""
{BLUE}                               .         ..  .                            .                      .       .     
                              ..   .                                    ..  .     .   ..            
                                                   .   .    .                   
                                      . .                                  .   . .      
                          .                     ..    .              . .     .  .          
      .       .   .            .          .      .                    .       . 
          .           /.         .      .               /     .                  
               ... .%#//%.    ..                   .  ##//%*   .              . 
     .       ..*#/////////%,       .                %#/////////%       .      . 
..          .  %(///////////%*.           .       %#///////////%*    .          
   .       .#  .%#////////////%/              ..%(////////////%.   /*   .       
 .. .  (#////#% ..%#////////////%(          ..%(////////////%.   (%/////%  .    
 .  .  .%//////(%.  %#////////////%(       .%/////////////%.   #%//////#/       
  ..   /#////////(%   %#////////////%#   ,%/////////////%,   ##////////(#    .  
    . (#////////////%. .%(///////////%*  %////////////%, . %#///////////(%. .  . 
 .      ##////////////%. .%(/////////%*  %//////////%*...%#///////////(%.       
      ..  ##////////////%, /#////////%*  %/////////%  .%(///////////(%. ..      
 .      ..  #%///////////#%/#////////%,  %/////////% (%///////////(%          ..
             .(%///////#%. /#////////%,  %/////////%   /%///////#% ..        . .
                (%///#%    /#////////%,  %/////////%     /%///#% .      .  .    
          .       /%#      /#////////%.  %/////////%       /%%       .  .       
    .  .                   /#////////%.  %/////////%     .               .      
     .                     /#////////%.  %/////////%.      .   .  .    .      . 
. .                .  .    /#////////%.  %/////////%     .  ..                   
.  .  .     ..  .. .. .    /#////////%.  %/////////%                        .   
     .       ..            /#////////%.  %/////////%              .             
. . ....                  ./#////////%. .%/////////%           ...    .      .  
                 .    .    /#////////%.  %/////////%          ..                
  . ...  .           ..    *%////////%.  ##///////(#..  ...               ..  . 
 ..             ..         ..%%%%%%%%  . .#%%%%%%%(           . .    .     .    
    .      .       .         . ..%            #,        .      .                
  .    .   ..   ..   .       .   %            #,.                    .          
   . .   .                .      %        .   #, .                              
         ..          .          .%.           #,        .             .         
    .    .  .   .     . ..  .   .% ..      .. #,     .                          
{RESET}

{BOLD}{YELLOW}                               AlignAIR CLI Tool{RESET}
{CYAN}                             ========================================{RESET}
"""
    print(art)
    print(f"{YELLOW}                               Welcome to the AlignAIR CLI!{RESET}")
    print(f"{CYAN}                             ========================================{RESET}")
def display_system_stats():
    """Display various system statistics in a pretty ASCII table."""

    # Number of processes
    number_of_processes = len(psutil.pids())

    # Free RAM in gigabytes
    free_ram_gb = psutil.virtual_memory().available / (1024 ** 3)

    # TensorFlow device info
    gpu_devices = tf.config.list_physical_devices('GPU')
    cpu_devices = tf.config.list_physical_devices('CPU')
    gpu_support = "Yes" if gpu_devices else "No"

    # Prepare data for the ASCII table
    table_data = [
        ["Number of Processes", number_of_processes],
        ["Free RAM (GB)", f"{free_ram_gb:.2f}"],
        ["TensorFlow GPU Support", gpu_support],
        ["GPU Devices Detected", ", ".join(d.name for d in gpu_devices) if gpu_devices else "None"],
        ["CPU Devices Detected", ", ".join(d.name for d in cpu_devices) if cpu_devices else "None"],
        ["Operating System", platform.platform()],
        ["Python Version", platform.python_version()]
    ]

    def print_ascii_table(data, title=None):
        """Helper to print a list of [key, value] rows in a pretty ASCII table."""
        # Calculate column widths
        left_col_width = max(len(str(row[0])) for row in data)
        right_col_width = max(len(str(row[1])) for row in data)

        total_width = left_col_width + right_col_width + 7  # padding for separators

        # Print top border
        print("+" + "-" * (total_width - 2) + "+")

        if title:
            # Center the title within the table width
            print("| " + title.center(total_width - 4) + " |")
            print("+" + "-" * (total_width - 2) + "+")

        # Print each row
        for key, val in data:
            left_cell = str(key).ljust(left_col_width)
            right_cell = str(val).ljust(right_col_width)
            print(f"| {left_cell} | {right_cell} |")

        # Print bottom border
        print("+" + "-" * (total_width - 2) + "+")

    # Print the ASCII table
    print_ascii_table(table_data, title="System Statistics")

display_header()
display_system_stats()

def parse_arguments():
    parser = argparse.ArgumentParser(description='AlignAIR Model Prediction')
    parser.add_argument('--mode', type=str, default='cli', choices=['cli', 'yaml', 'interactive'],
                        help='Mode of input: cli, yaml, interactive')
    parser.add_argument('--config_file', type=str, help='Path to YAML configuration file')
    parser.add_argument('--model_checkpoint', type=str, help='Path to saved AlignAIR weights')
    parser.add_argument('--save_path', type=str, help='Where to save the alignment')
    parser.add_argument('--chain_type', type=str, help='heavy / light')
    parser.add_argument('--sequences', type=str,
                        help='Path to csv/tsv/fasta file with sequences in a column called "sequence"')
    parser.add_argument('--lambda_data_config', type=str, default='D', help='Path to lambda chain data config')
    parser.add_argument('--kappa_data_config', type=str, default='D', help='Path to kappa chain data config')
    parser.add_argument('--heavy_data_config', type=str, default='D', help='Path to heavy chain data config')
    parser.add_argument('--max_input_size', type=int, default=576, help='Maximum model input size')
    parser.add_argument('--batch_size', type=int, default=2048, help='The Batch Size for The Model Prediction')
    parser.add_argument('--v_allele_threshold', type=float, default=0.1, help='Percentage for V allele assignment')
    parser.add_argument('--d_allele_threshold', type=float, default=0.1, help='Percentage for D allele assignment')
    parser.add_argument('--j_allele_threshold', type=float, default=0.1, help='Percentage for J allele assignment')
    parser.add_argument('--v_cap', type=int, default=3, help='Cap for V allele calls')
    parser.add_argument('--d_cap', type=int, default=3, help='Cap for D allele calls')
    parser.add_argument('--j_cap', type=int, default=3, help='Cap for J allele calls')
    parser.add_argument('--translate_to_asc', action='store_true', help='Translate names back to ASCs names from IMGT')
    parser.add_argument('--fix_orientation', type=bool, default=True,
                        help='Adds a preprocessing step that tests/fixes orientation if reversed/complement')
    parser.add_argument('--custom_orientation_pipeline_path', type=str, default=None,
                        help='Path to a custom orientation model created for a custom reference')
    parser.add_argument('--custom_genotype', type=str, default=None, help='Path to a custom genotype yaml file')
    parser.add_argument('--save_predict_object', action='store_true',
                        help='Save the predict object (Warning this can be large)')
    parser.add_argument('--airr_format', action='store_true', help='Adds a step to format the results to AIRR format')
    # parameters for the model yaml, if specified, changes loading to a fine-tuned model with different head sizes
    parser.add_argument('--finetuned_model_params_yaml', type=str, default=None,
                        help='Path to a yaml file with the parameters of a fine-tuned model')
    return parser.parse_args()


def load_yaml_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return Args(**config)


def interactive_mode():
    """Prompt-based config in 'interactive' mode."""
    config = {
        'model_checkpoint': questionary.text("Path to saved AlignAIR weights:").ask(),
        'save_path': questionary.text("Where to save the alignment:").ask(),
        'chain_type': questionary.select("Chain type:", choices=['heavy', 'light']).ask(),
        'sequences': questionary.text("Path to csv/tsv/fasta file with sequences in a column called 'sequence':").ask(),
        'lambda_data_config': questionary.text("Path to lambda chain data config:", default='D').ask(),
        'kappa_data_config': questionary.text("Path to kappa chain data config:", default='D').ask(),
        'heavy_data_config': questionary.text("Path to heavy chain data config:", default='D').ask(),
        'max_input_size': int(questionary.text("Maximum model input size:", default='576').ask()),
        'batch_size': int(questionary.text("Batch size for the model prediction:", default='2048').ask()),
        'v_allele_threshold': float(questionary.text("Threshold for V allele prediction:", default='0.75').ask()),
        'd_allele_threshold': float(questionary.text("Threshold for D allele prediction:", default='0.3').ask()),
        'j_allele_threshold': float(questionary.text("Threshold for J allele prediction:", default='0.8').ask()),
        'v_cap': int(questionary.text("Cap for V allele calls:", default='3').ask()),
        'd_cap': int(questionary.text("Cap for D allele calls:", default='3').ask()),
        'j_cap': int(questionary.text("Cap for J allele calls:", default='3').ask()),
        'translate_to_asc': questionary.confirm("Translate names back to ASCs names from IMGT?").ask(),
        'fix_orientation': questionary.confirm("Fix DNA orientation if reversed?").ask(),
        'airr_format': questionary.confirm("Format the results to AIRR format?").ask(),
        'custom_orientation_pipeline_path': questionary.text("Path to a custom orientation model:", default='').ask(),
        'custom_genotype': questionary.text("Path to a custom genotype yaml file:", default='').ask(),
        'save_predict_object': questionary.confirm("Save the predict object?").ask(),
        'finetuned_model_params_yaml': questionary.text("Path to a yaml file with fine-tuned model parameters:",
                                                        default='').ask()
    }
    return Args(**config)


def run_pipeline(predict_object, steps):
    for step in steps:
        predict_object = step.execute(predict_object)


def execute_pipeline(config):
    logger = logging.getLogger('PipelineLogger')
    predict_object = PredictObject(config, logger=logger)
    Step.set_logger(logger=logger)

    steps = [
        ConfigLoadStep("Load Config"),
        FileNameExtractionStep('Get File Name'),
        FileSampleCounterStep('Count Samples in File'),
        ModelLoadingStep('Load Models'),
        BatchProcessingStep("Process and Predict Batches"),
        CleanAndArrangeStep("Clean Up Raw Prediction"),
        SegmentCorrectionStep("Correct Segmentations"),
        MaxLikelihoodPercentageThresholdApplicationStep("Apply Max Likelihood Threshold to Distill Assignments"),
        AlleleAlignmentStep("Align Predicted Segments with Germline")
    ]

    # Decide final steps based on AIRR or not
    if config.airr_format:
        steps.append(AIRRFinalizationStep("Finalize Results"))
    else:
        steps.append(TranslationStep("Translate ASC's to IMGT Alleles"))
        steps.append(FinalizationStep("Finalize Results"))

    run_pipeline(predict_object, steps)
    logger.info("Pipeline execution complete.")


def show_menu():
    """Fallback menu if user runs with no arguments or explicitly chooses interactive mode."""
    display_header()
    menu = f"""
{BOLD}{CYAN}========================================{RESET}
{BOLD}{YELLOW} Please choose an option below:{RESET}
{CYAN}========================================{RESET}
1. Run Model in CLI Mode
2. Run Model with YAML Configuration
3. Run Model in Interactive Mode
4. Exit
"""
    while True:
        print(menu)
        choice = input(f"{BLUE}Your choice (1-4): {RESET}").strip()

        if choice == '1':
            # Prompt the user to enter CLI arguments as a string
            cli_command = input(
                f"{BLUE}Enter CLI arguments (e.g., --model_checkpoint path/to/checkpoint --save_path path/to/save ...): {RESET}").strip()
            if cli_command:
                # Use shlex.split to parse the command-line string into a list of arguments
                sys.argv = [sys.argv[0]] + shlex.split(cli_command)
                config = parse_arguments()
                execute_pipeline(config)
            else:
                print(f"{RED}No arguments provided. Returning to menu.{RESET}")
        elif choice == '2':
            config_file = input(f"{BLUE}Enter the path to the YAML config file: {RESET}").strip()
            config = load_yaml_config(config_file)
            execute_pipeline(config)
        elif choice == '3':
            config = interactive_mode()
            execute_pipeline(config)
        elif choice == '4':
            print(f"{GREEN}Exiting AlignAIR. Goodbye!{RESET}")
            break
        else:
            print(f"{RED}Invalid choice. Please select a valid option (1-4).{RESET}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # First, parse arguments.
    # The user can set --mode=cli or --mode=yaml or --mode=interactive.
    args = parse_arguments()

    # If user didn't provide any arguments besides the default,
    # or if "mode" is "interactive", we can show the menu or interactive flow.
    if len(sys.argv) == 1:
        # Means user just typed 'python app.py' with no arguments
        show_menu()
    else:
        if args.mode == 'interactive':
            # Let user fill out config with questionary
            config = interactive_mode()
            execute_pipeline(config)
        elif args.mode == 'yaml':
            # Must have --config_file specified
            if not args.config_file:
                print(f"{RED}No config_file specified. Please use --config_file /path/to/config.yaml{RESET}")
                sys.exit(1)
            config = load_yaml_config(args.config_file)
            execute_pipeline(config)
        else:
            # Otherwise, "cli" mode
            config = args  # directly use parsed arguments
            execute_pipeline(config)

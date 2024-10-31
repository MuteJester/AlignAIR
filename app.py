# app.py
import argparse
import logging
import os
import shlex
import sys
import yaml
import questionary
import tensorflow as tf
from AlignAIR.PostProcessing.Steps.allele_threshold_step import MaxLikelihoodPercentageThresholdApplicationStep, ConfidenceMethodThresholdApplicationStep
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
{BLUE}.         ..  .                            .                      .       .     
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
         . .      .  .      .   .      .  . .   .       .               .  .. . {RESET}

{BOLD}{YELLOW}TensorFlow Model CLI Tool{RESET}
{CYAN}========================================{RESET}
"""
    print(art)
    print(f"{YELLOW}Welcome to the TensorFlow Model CLI!{RESET}")
    print(f"{CYAN}========================================{RESET}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='AlignAIR Model Prediction')
    parser.add_argument('--mode', type=str, default='cli', choices=['cli', 'yaml', 'interactive'],
                        help='Mode of input: cli, yaml, interactive')
    parser.add_argument('--config_file', type=str, help='Path to YAML configuration file')
    parser.add_argument('--model_checkpoint', type=str, help='Path to saved AlignAIR weights', required=True)
    parser.add_argument('--save_path', type=str, help='Where to save the alignment', required=True)
    parser.add_argument('--chain_type', type=str, help='heavy / light', required=True)
    parser.add_argument('--sequences', type=str, help='Path to csv/tsv/fasta file with sequences in a column called "sequence"', required=True)
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
    parser.add_argument('--fix_orientation', type=bool, default=True, help='Fix DNA orientation if reversed')
    parser.add_argument('--custom_orientation_pipeline_path', type=str, default=None, help='Path to custom orientation model')

    return parser.parse_args()

def load_yaml_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return Args(**config)

def interactive_mode():
    config = {
        'model_checkpoint': questionary.text("Path to saved AlignAIR weights:").ask(),
        'save_path': questionary.text("Where to save the alignment:").ask(),
        'chain_type': questionary.select("Chain type:", choices=['heavy', 'light']).ask(),
        'sequences': questionary.text("Path to csv/tsv file with sequences in a column called 'sequence':").ask(),
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
        'custom_orientation_pipeline_path': questionary.text("Path to a custom orientation model:", default='').ask(),
    }
    return Args(**config)

def run_pipeline(predict_object, steps):
    for step in steps:
        predict_object = step.execute(predict_object)

def execute_pipeline(config):
    logger = logging.getLogger('PipelineLogger')
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

def show_menu():
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
            # Prompt the user to enter command-line arguments as a string
            cli_command = input(
                f"{BLUE}Enter CLI arguments (e.g., --model_checkpoint path/to/checkpoint --save_path path/to/save ...): {RESET}").strip()
            if cli_command:
                # Use shlex.split to parse the command-line string into a list of arguments
                sys.argv = [sys.argv[0]] + shlex.split(cli_command)  # Replace sys.argv with the new command
                config = parse_arguments()
                print(config)
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
            print(f"{GREEN}Exiting the AlignAIR. Goodbye!{RESET}")
            break
        else:
            print(f"{RED}Invalid choice. Please select a valid option (1-4).{RESET}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    show_menu()

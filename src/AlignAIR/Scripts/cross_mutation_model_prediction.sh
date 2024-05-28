#!/API/bash

# Usage function to display help
usage() {
    echo "Usage: $0 -m model1,model2,model3 -c csv1,csv2,csv3 -s save_path"
    echo "  -m: Comma-separated list of model checkpoint paths"
    echo "  -c: Comma-separated list of CSV file paths"
    echo "  -s: Path to save the prediction results"
    exit 1
}

# Parse command line options
while getopts ":m:c:s:" opt; do
    case $opt in
        m) model_checkpoints=(${OPTARG//,/ });;
        c) csv_files=(${OPTARG//,/ });;
        s) save_path=$OPTARG;;
        \?) echo "Invalid option -$OPTARG" >&2
            usage;;
    esac
done

# Check if all arguments are provided
if [ -z "${model_checkpoints}" ] || [ -z "${csv_files}" ] || [ -z "${save_path}" ]; then
    usage
fi

# Specify the Python executable path
python_executable="/home/bcrlab/thomas/anaconda3/bin/python"
# Specify the path to the AlignAIRR_Predict.py script
predict_script="/home/bcrlab/thomas/AlignAIRR/Scripts/AlignAIRR_Predict.py"

# Loop over each model checkpoint
for model_checkpoint in "${model_checkpoints[@]}"; do
    echo "Processing model checkpoint: $model_checkpoint"

    # Loop over each CSV file
    for csv_file in "${csv_files[@]}"; do
        echo "Generating predictions for $csv_file using model $model_checkpoint"

        # Call Python script to generate predictions using the specified Python executable
        $python_executable $predict_script --model_checkpoint "$model_checkpoint" --csv_file "$csv_file" --save_path "$save_path"
    done
done

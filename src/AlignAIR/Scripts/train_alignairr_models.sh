#!/API/bash

# Define the models
models=("S5F") #"Uniform" "S5F_60" "S5F_Opposite")
python_executable="/home/bcrlab/thomas/anaconda3/bin/python"
predict_script="/home/bcrlab/thomas/AlignAIRR/Scripts/Train_HeavyChain_AlignAIRR.py"

# Iterate over each model
for model in "${models[@]}"; do
    echo "Running AlignAIRR Training for model: $model"
    # Set the MODEL environment variable
    export MODEL="$model"
    # Run the Python script
    $python_executable $predict_script

done

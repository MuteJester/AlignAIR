#!/bin/bash

# Define the models
models=("Uniform" "S5F" "S5F_60" "S5F_Opposite")
python_executable="/home/bcrlab/thomas/anaconda3/bin/python"
predict_script="/home/bcrlab/thomas/AlignAIRR/Scripts/Generate_Train_Dataset.py"

# Iterate over each model
for model in "${models[@]}"; do
    echo "Running simulation for model: $model"
    # Set the MODEL environment variable
    export MODEL="$model"
    # Run the Python script
    $python_executable $predict_script

done

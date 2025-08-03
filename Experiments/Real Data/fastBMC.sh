#!/bin/bash


# File paths
input_file="input.csv"
temp_output="output.csv"
results_file="results_fastBMC.csv"

# Initialise le fichier de résultats
echo "num_features,time,auc" > "$results_file"

for nb in $(seq 50 50 800); do
    echo "Processing with $nb features..."

    # Filter and keep only the most varying features
    python3 filter_features.py "$input_file" "$temp_output" "$nb"

    # Compute and extract the time and performance
    output=$(python3 run_fastBMC_time_auc.py "$temp_output")

    # Extract values
    time=$(echo "$output" | cut -d',' -f1)
    auc=$(echo "$output" | cut -d',' -f2)

    # Store the values in results_fastBMC.csv
    echo "$nb,$time,$auc" >> "$results_file"

    # Remove the temporary dataset
    rm -f "$temp_output"
done



#!/bin/bash


# File paths
input_dir="Simulated_data/"
results_file="results_fastBMC.csv"

# Initialise le fichier de résultats
echo "dataset,time,auc,acc, f1" > "$results_file"

for file in "$input_dir"*; do
    if [ -f "$file" ]; then
        # Process each file 
        echo "Processing $file"

        # Compute and extract the time and performance
        output=$(python3 run_fastBMC_simulated_data.py "$file")

        # Extract values
        time=$(echo "$output" | cut -d',' -f1)
        auc=$(echo "$output" | cut -d',' -f2)
        acc=$(echo "$output" | cut -d',' -f3)
        f1=$(echo "$output" | cut -d',' -f4)

        # Store the values in results_fastBMC.csv
        echo "$file,$time,$auc,$acc,$f1" >> "$results_file"
        
    fi

done

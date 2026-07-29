#!/bin/bash

thresholds=(0 0.064 0.074 0.08 0.085 0.091)
seeds=(43 44 3407)
LOG_DIR="logs"

for t in "${thresholds[@]}"; do
    for s in "${seeds[@]}"; do
        # Format threshold for naming (e.g., 0.032 -> 032)
        t_suffix=$(echo $t | sed 's/0\.//')
        batch_name="5o_nano_131_online_p${t_suffix}_seed_${s}"
        
        # Define the full path to the expected log folder
        current_log_path="${LOG_DIR}/${batch_name}"

        # SAFETY CHECK: Only run if the directory DOES NOT exist
        if [ ! -d "$current_log_path" ]; then
            echo "🚀 Starting experiment: $batch_name"
            
            python scripts/batch_run_online.py \
                --threshold "$t" \
                --batch_name "$batch_name" \
                --seed "$s" \
                --max_jobs 50 \
                --config config/CPnav_5_nano_131.yaml
        else
            echo "⏩ Skipping: $batch_name (Folder already exists at $current_log_path)"
        fi

    done
done
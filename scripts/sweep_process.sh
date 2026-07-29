#!/bin/bash

set -euo pipefail

LOG_DIR="logs/mp3d"

# Ensure the log directory exists before starting
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Directory $LOG_DIR does not exist."
    exit 1
fi

# Iterate through every subdirectory inside LOG_DIR
for current_log_path in "$LOG_DIR"/*/; do
    # Remove trailing slash for cleaner logging/processing
    current_log_path="${current_log_path%/}"

    echo "Processing: $current_log_path"

    # 1. Run successful_images.sh if goal_image dir is missing
    if [ ! -d "$current_log_path/goal_image" ]; then
        bash scripts/successful_images.sh "$current_log_path"
    fi

    # 2. Run false positive check if fp dir is missing
    if [ ! -d "$current_log_path/false_positive" ]; then
        python scripts/check_false_positive.py --folder "$current_log_path/goal_image"
        mv "$current_log_path/goal_image/fp" "$current_log_path/false_positive"
        mv "$current_log_path/goal_image/false_positive.txt" "$current_log_path"
    fi

    # 3. Process results if results.csv is missing
    # Note: Added spaces inside [ ] to prevent syntax errors
    if [ ! -f "$current_log_path/results.csv" ]; then
        python scripts/process_results.py --exp_dir "$current_log_path"
    fi
done

echo "All folders processed."
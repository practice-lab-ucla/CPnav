import os
import json
import pandas as pd
from pathlib import Path


def calculate_path_length(path_list, grid_size=0.5):
    """Calculates Euclidean distance between grid points and scales by grid size."""
    if not path_list or len(path_list) < 2:
        return 0.0
    
    total_dist = 0.0
    for i in range(len(path_list) - 1):
        p1 = path_list[i]
        p2 = path_list[i+1]
        # Euclidean distance between grid indices
        dist = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
        total_dist += dist
    
    return total_dist * grid_size

def process_experiments(root_dir):
    data_rows = []
    root_path = Path(root_dir)

    # Iterate through each episode folder
    for episode_folder in root_path.glob("episode_*"):
        # Extract episode number from folder name: episode_{num}_{time}
        parts = episode_folder.name.split('_')
        episode_num = parts[1] if len(parts) > 1 else episode_folder.name

        # Initialize default values
        row = {
            "episode_number": episode_num,
            "goal": None,
            "step_count": None,
            "success": False,
            "total_distance_traversed": None,
            "geodesic_distance": None,
            "offline_score": None,
            "best_path_step_count": None,
            "estimated_best_path_length": None
        }

        # 1. Determine Success/Failure and load terminal summary
        term_success = episode_folder / "terminal" / "terminal_summary.json"
        term_failure = episode_folder / "terminal" / "failure_summary.json"
        
        summary_path = None
        if term_success.exists():
            summary_path = term_success
            row["success"] = True
        elif term_failure.exists():
            summary_path = term_failure
            row["success"] = False

        if summary_path:
            with open(summary_path, 'r') as f:
                ts = json.load(f)
                row["goal"] = ts.get("goal")
                row["step_count"] = ts.get("step_count")
                row["total_distance_traversed"] = ts.get("total_distance_traversed")
                row["geodesic_distance"] = ts.get("geodesic_distance") # Empty if failure

        # 2. Load offline summary
        offline_path = episode_folder / "offline" / "offline_summary.json"
        if offline_path.exists():
            with open(offline_path, 'r') as f:
                off = json.load(f)
                best_path = off.get("best_path", [])
                row["offline_score"] = off.get("score")
                row["best_path_step_count"] = len(best_path)
                row["estimated_best_path_length"] = calculate_path_length(best_path, 0.5)

        data_rows.append(row)

    # Create DataFrame and sort by episode number
    df = pd.DataFrame(data_rows)
    df['episode_number'] = pd.to_numeric(df['episode_number'], errors='ignore')
    df = df.sort_values('episode_number')
    
    return df

# Usage
root_directory = "logs/batch_20260214_000158" # Change this to your actual path
results_df = process_experiments(root_directory)
results_df.to_csv("experiment_results_summary.csv", index=False)

print("CSV generated successfully: experiment_results_summary.csv")
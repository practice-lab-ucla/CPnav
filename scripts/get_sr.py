import pandas as pd
import argparse
import sys

def calculate_rates(file_path):
    try:
        # Load the data
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Helper function to check for "True" values
    # Handles both boolean (True) and string ("True", "TRUE", " true ") formats
    def is_true(val):
        if pd.isna(val):
            return False
        return str(val).strip().upper() == 'TRUE'

    # Ensure required columns exist
    required_cols = ['success', 'FP', 'step_count']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain the columns: {', '.join(required_cols)}")
        return

    # Identify 'Real Success'
    # Real Success = (success is True) AND (FP is NOT True)
    df['is_success'] = df['success'].apply(is_true)
    df['is_fp'] = df['FP'].apply(is_true)
    df['is_real_success'] = df['is_success'] & ~df['is_fp']
    
    # Define the step count limits to check
    limits = [30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
    
    # Count total attempts (rows where success data is present)
    total_episodes = df['success'].dropna().shape[0]
    
    # Print results to terminal
    print(f"\nAnalyzing File: {file_path}")
    print(f"Total Valid Episodes: {total_episodes}")
    print("=" * 55)
    print(f"{'Step Limit':<12} | {'Real Successes':<15} | {'Success Rate (%)':<15}")
    print("-" * 55)
    
    for L in limits:
        # Count real successes that happened within the step limit L
        count = (df['is_real_success'] & (df['step_count'] <= L)).sum()
        rate = (count / total_episodes * 100) if total_episodes > 0 else 0
        print(f"{L:<12} | {count:<15} | {rate:<15.2f}")
    
    print("=" * 55)
    print("Note: Real Success = (Success: True) AND (FP: NOT True)\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate success rates from results CSV.')
    parser.add_argument('csv_file', nargs='?', default='results.csv', 
                        help='Path to the results CSV file (default: results.csv)')
    args = parser.parse_args()
    
    calculate_rates(args.csv_file)
import os
import ast
import pandas as pd


log_dir = "logs/mp3d"

for d in os.listdir(log_dir):
    result_path = os.path.join(log_dir, d, "results.csv")
    df = pd.read_csv(result_path)

    fp_path = os.path.join(log_dir, d, "false_positive.txt")

    with open(fp_path, "r") as f:
        fp_list = ast.literal_eval(f.read())

    # 1. Add 'False positive' column
    df['False positive'] = df['episode_number'].isin(fp_list)

    # 2. Add 'true_success' column (must be success == True AND False positive == False)
    df['true_success'] = df['success'] & (~df['False positive'])

    # 3. Set offline_score to 0 where true_success is False (This also covers False positives)
    df.loc[~df['true_success'], 'offline_score'] = 0

    # Save to a new CSV
    df.to_csv(f'{d}_result.csv', index=False)
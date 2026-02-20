import os
import time
import numpy as np
import subprocess
from datetime import datetime

# --- Configuration ---
SCRIPT = "scripts/run_online.py"

import argparse

parser = argparse.ArgumentParser(description="run online experiment in batch")

parser.add_argument(
    "--config",
    type=str,
    default="config/CPnav_5_nano.yaml",
    help="path to config file",
)

parser.add_argument(
    "--max_jobs",
    type=int,
    default=15,
    help="maximum number of jobs to run in parallel",
)

parser.add_argument(
    "--seed",
    type=int,
    default=100,
    help="seed"
)

parser.add_argument(
    "--threshold",
    type=float,
    default=0.0,
    help="score threshold to filter out backtrack actions"
)

parser.add_argument(
    "--n_exp",
    type=int,
    default=100,
    help="number of experiments to run"
)

parser.add_argument(
    "--batch_name",
    type=str,
    default="",
    help="name of the batch experiment folder"
)

parser.add_argument(
    "--mode",
    type=str,
    default="cp_nav",  # Updated from 100 to a valid default
    choices=["cp_nav", "simple_set", "prompt_set", "ensemble_set"],
    help="Navigation mode for the agent (default: cp_nav)",
)

parser.add_argument(
    "--alpha",
    type=float,
    default=0.3,  # Updated from 100 to a valid default
    help="Threshold used for constructing prediction set, only used if mode = simple_set or ensemble_set",
)

args = parser.parse_args()

MAX_JOBS = args.max_jobs
SEED = args.seed
N_EXPERIMENTS = args.n_exp
THR = args.threshold
CONFIG = args.config

np.random.seed(SEED)
EPISODES = np.random.choice(np.arange(0, 1000), size=N_EXPERIMENTS, replace=False).tolist()

# Create timestamped log directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if (args.batch_name == "") or (args.batch_name is None):
    log_path = f"logs/batch_{timestamp}"
else:
    log_path = f"logs/{args.batch_name}"

term_path = os.path.join(log_path, "terminal")
os.makedirs(term_path, exist_ok=True)

def run_experiments():
    processes = []
    episodes_to_run = EPISODES.copy()
    
    print(f"Starting batch experiment: {len(EPISODES)} episodes, {MAX_JOBS} at a time.")

    while episodes_to_run or processes:
        # 1. Fill up the queue if we have space and episodes left
        while len(processes) < MAX_JOBS and episodes_to_run:
            ep = episodes_to_run.pop(0)
            log_file = os.path.join(term_path, f"episode_{ep}.log")
            
            print(f"Launching Episode {ep}...")
            
            # Start process (non-blocking)
            f = open(log_file, "w")
            proc = subprocess.Popen(
                ["python", SCRIPT, 
                    "--config", CONFIG, 
                    "--log_dir", log_path, 
                    "--episode", str(ep),
                    "--threshold", str(THR),
                    "--mode", str(args.mode),
                    "--alpha", str(args.alpha)
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            processes.append({"proc": proc, "episode": ep, "file": f})

        # 2. Check for finished processes
        for p_dict in processes[:]:
            retcode = p_dict["proc"].poll()
            if retcode is not None:  # Process finished
                p_dict["file"].close()
                processes.remove(p_dict)
                status = "Success" if retcode == 0 else f"Failed (code {retcode})"
                print(f"Finished Episode {p_dict['episode']} - Status: {status}")

        # 3. Small sleep to prevent CPU spiking while monitoring
        time.sleep(1)

    print("\nAll experiments completed.")

if __name__ == "__main__":
    start_time = time.time()
    try:
        run_experiments()
    except KeyboardInterrupt:
        print("\nReceived KeyboardInterrupt — stopping early.")
    finally:
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"Total time elapsed: {hours}h {minutes}m {seconds}s")

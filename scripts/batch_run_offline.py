import os
import time
import yaml
import numpy as np
import subprocess
from datetime import datetime

from CPnav.env import ObjectNavEnv

import argparse

parser = argparse.ArgumentParser(description="run offline experiment in batch")

parser.add_argument(
    "--config",
    type=str,
    default="config/CPnav_5_nano.yaml",
    help="path to config file",
)

parser.add_argument(
    "--max_jobs",
    type=int,
    default=10,
    help="maximum number of jobs to run in parallel",
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="seed"
)

parser.add_argument(
    "--n_exp",
    type=int,
    default=300,
    help="number of experiments to run"
)

parser.add_argument(
    "--batch_name",
    type=str,
    default="",
    help="name of the batch experiment folder"
)

parser.add_argument(
    "--forbidden_range",
    type=int,
    nargs=2,
    default=[200, 222],
)

parser.add_argument(
    "--log_folder",
    type=str,
    default="logs/mp3d",
    help="Only sample episodes whose goal object_category is in this list",
)

parser.add_argument(
    "--goal_categories",
    type=str,
    nargs="+",
    default=["chair", "table", "plant", "sofa", "bed", "toilet", "tv_monitor"],
    help="Only sample episodes whose goal object_category is in this list",
)

args = parser.parse_args()

MAX_JOBS = args.max_jobs
SEED = args.seed
N_EXPERIMENTS = args.n_exp
CONFIG = args.config
SCRIPT = "scripts/run_offline.py"
forbidden_low, forbidden_high = args.forbidden_range
goal_categories = set(args.goal_categories)

with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)
env = ObjectNavEnv(cfg)

all_episodes_filtered = [
    i for i, ep in enumerate(env.all_episodes)
    if ep["object_category"] in goal_categories and not (forbidden_low < i < forbidden_high)
]
if N_EXPERIMENTS > len(all_episodes_filtered):
    raise ValueError(
        f"--n_exp {N_EXPERIMENTS} exceeds the {len(all_episodes_filtered)} episodes "
        f"available for goal categories {sorted(goal_categories)} (after excluding the forbidden range)."
    )
np.random.seed(SEED)
EPISODES = np.random.choice(all_episodes_filtered, size=N_EXPERIMENTS, replace=False).tolist()

# Create timestamped log directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.batch_name == "" or args.batch_name is None:
    log_path = f"{args.log_folder}/batch_{timestamp}"
else:
    log_path = f"{args.log_folder}/{args.batch_name}"

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
                ["python", SCRIPT, "--config", CONFIG, "--log_dir", log_path, "--episode", str(ep)],
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

# CPNAV

**CPNAV** is a **Conformal Prediction–augmented Navigation** framework for object-goal search in unseen indoor environments. It uses RGB-D observations to propose collision-free actions, scores them with a vision-language model, and applies conformal prediction to filter unreliable decisions before executing actions through a tree-search policy with backtracking.

![CPNAV Overview](Images/CPnav.png)

## Installation

This section includes all setup steps: creating the conda environment, setting up the OpenAI API key, and downloading the required datasets.

### Step 1: Create the Conda Environment

Navigate to the directory you want to install the project:

```bash
conda env create -f environment.yml
conda activate CPnav
```

This installs all dependencies, including `habitat-sim`.

### Step 2: Set Up the OpenAI API Key

CPNAV uses the OpenAI API for VLM-based action scoring and goal checking.

Create a `.env` file in the root `CPnav` directory:

```bash
touch .env
```

Open the `.env` file and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

For example:

```bash
OPENAI_API_KEY=sk-...
```

Do not include spaces around the `=` sign. Do not commit your `.env` file to GitHub. Make sure `.env` is listed in your `.gitignore` file:


### Dataset Setup Explanation for Step 3 to 6

This project follows a data setup similar to the one used in the [VLMnav repository](https://github.com/Jirl-upenn/VLMnav/blob/main/readme.md) for the paper [*End-to-End Navigation with Vision-Language Models: Transforming Spatial Reasoning into Question-Answering*](https://arxiv.org/pdf/2411.05755).

By default, CPNAV uses HM3D scenes with ObjectNav HM3D v2 episodes (Steps 3–5). The configs also include a dataset switch for MP3D/Matterport3D (Step 6), loaded from `data/datasets/scene_datasets/mp3d` with episodes under `data/datasets/objectnav/mp3d/v1`.

### Step 3: Download the Habitat-Matterport 3D Research Dataset

You need access to the Habitat-Matterport 3D Research Dataset:

[https://matterport.com/habitat-matterport-3d-research-dataset](https://matterport.com/habitat-matterport-3d-research-dataset)

After receiving access, generate a Matterport API token:

1. Go to [https://my.matterport.com/settings/account/devtools](https://my.matterport.com/settings/account/devtools)
2. Generate an API token.
3. Use the API token ID as your username.
4. Use the API token secret as your password.

Download the HM3D validation scenes:

```bash
python -m habitat_sim.utils.datasets_download \
  --username <api-token-id> \
  --password <api-token-secret> \
  --uids hm3d_val_v0.2
```

### Step 4: Download the ObjectNav Episode Dataset

Download the ObjectNav HM3D v2 episode dataset from:

[https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip)

### Step 5: Organize the Downloaded Dataset

Organize both datasets under `data/` exactly as below — these are the paths the
configs (`config/CPnav_5_nano_131*.yaml`) read from:

```text
data/
├── versioned_data/                          # HM3D scenes
│   └── hm3d-0.2/
│       └── hm3d/
│           ├── hm3d_annotated_basis.scene_dataset_config.json
│           └── val/
│               └── <scene_id>/
│                   ├── <scene_id>.basis.glb
│                   ├── <scene_id>.basis.navmesh
│                   ├── <scene_id>.semantic.glb
│                   └── <scene_id>.semantic.txt
│
└── datasets/
    ├── objectnav_hm3d_v2/                    # HM3D ObjectNav v2 episodes
    │   └── val/
    │       ├── val.json.gz
    │       └── content/*.json.gz
    │
    ├── scene_datasets/
    │   └── mp3d/                             # MP3D (Matterport3D) scenes
    │       └── <scene_id>/
    │           └── <scene_id>.glb
    │
    └── objectnav/
        └── mp3d/
            └── v1/                           # MP3D ObjectNav episodes
                └── val/
                    ├── val.json.gz
                    └── content/*.json.gz
```

Paths per dataset (from the configs):

| dataset | scenes                              | episodes                          | config                             |
| ------- | ----------------------------------- | --------------------------------- | ---------------------------------- |
| HM3D    | `data/versioned_data/hm3d-0.2/hm3d` | `data/datasets/objectnav_hm3d_v2` | `config/CPnav_5_nano_131.yaml`     |
| MP3D    | `data/datasets/scene_datasets/mp3d` | `data/datasets/objectnav/mp3d/v1` | `config/CPnav_5_nano_131_mp3d.yaml`|

MP3D scenes are loaded directly from the `.glb` (no scene-dataset config), so
MP3D ships no `.navmesh` — the simulator recomputes the navmesh from geometry on
load. HM3D uses its bundled `.basis.navmesh`.

To switch a config between datasets, set the dataset name (`run.sh` picks the
right config for you, so you usually don't touch this):

```yaml
env_cfg:
  dataset:
    name: hm3d  # or mp3d
```

### Step 6: Download the MP3D (Matterport3D) Dataset

The steps above set up HM3D. For the MP3D experiments you also need the
Matterport3D scenes and the ObjectNav MP3D episodes.

**Option A — packaged with SG-Nav (simplest).** The
[SG-Nav repo](https://github.com/bagh2178/SG-Nav) documents an MP3D + ObjectNav
data setup; follow its data instructions and place the scenes under
`data/datasets/scene_datasets/mp3d/` and the episodes under
`data/datasets/objectnav/mp3d/v1/` as in the tree above.

**Option B — official sources.**

- **MP3D scenes:** request access to the Matterport3D dataset
  (<https://niessner.github.io/Matterport/>), then download the Habitat-ready
  scenes via the habitat downloader:

  ```bash
  python -m habitat_sim.utils.datasets_download --uids mp3d --data-path data/
  ```

  Place them so each scene is `data/datasets/scene_datasets/mp3d/<scene_id>/<scene_id>.glb`.
- **ObjectNav MP3D episodes:** the Habitat ObjectNav MP3D v1 episodes
  (`objectnav_mp3d_v1`, from the Habitat data releases) extracted to
  `data/datasets/objectnav/mp3d/v1/` (with `val/val.json.gz` and `val/content/`).

Only the 11 MP3D `val` scenes referenced by the evaluated episodes are needed.

## Usage

### Quickstart — reproduce the paper runs

The paper evaluates on a **fixed 200-episode, 6-category** set (chair, plant,
sofa, bed, toilet, tv_monitor) on **HM3D** (seed 43) and **MP3D** (seed 96).
CPnav's four internal baselines are the *same agent* under a different
prediction-set mode, so they all share one config per dataset — pick the method
by name, no per-method config file:

```bash
conda activate CPnav               # or set OPENAI_API_KEY in .env

bash run.sh <hm3d|mp3d> <method>   # one method, one dataset -> logs/<dataset>_<method>/
bash run_all.sh <hm3d|mp3d>        # all five methods, sequentially
```

| method     | flags                              | description                    |
| ---------- | ---------------------------------- | ------------------------------ |
| `cpnav`    | `--mode cp_nav --threshold 0.064`  | CPnav, conformal (main method) |
| `nocal`    | `--mode cp_nav --threshold 0`      | No Calibration                 |
| `simple`   | `--mode simple_set --alpha 0.3`    | Simple Set                     |
| `prompt`   | `--mode prompt_set`                | Prompt Set                     |
| `ensemble` | `--mode ensemble_set --alpha 0.3`  | Ensemble Set (~20 calls/step)  |

```bash
bash run.sh hm3d cpnav             # CPnav on HM3D
bash run.sh mp3d ensemble          # Ensemble baseline on MP3D
```

Only the two configs used for the paper runs live at `config/`
(`CPnav_5_nano_131.yaml` = HM3D, `CPnav_5_nano_131_mp3d.yaml` = MP3D). Ablation
configs (no-voxel, sensor-height, propose-if-all-explored) are under
`config/ablations/`.

### Manual commands

The sections below expose the underlying scripts for one-off runs and offline
calibration. This repository follows the workflow from the CPNav paper: first
run offline calibration/search to collect calibration scores, then run the
online implementation with the selected conformal threshold. The scores
gathered offline are bounded with the conformal prediction bound using a
human-specified risk threshold, and the resulting threshold is used by the
online policy. All commands should be run from the root `CPnav` directory after
activating the conda environment:

```bash
conda activate CPnav
```

### Command 1: Run Offline Calibration for One Episode

Use `scripts/run_offline.py` to run one offline calibration/search episode. This
records the online trajectory first, then refines the explored tree offline and
saves the offline score and best path under the log directory.

```bash
python scripts/run_offline.py \
  --config config/CPnav_5_nano_131.yaml \
  --log_dir logs/offline_single \
  --episode 100
```

Key arguments:

1. `--config`: Path to the experiment configuration file.
2. `--log_dir`: Directory where logs, images, JSON files, and summaries are saved.
3. `--episode`: Episode index to run.

### Command 2: Run Online Implementation for One Episode

Use `scripts/run_online.py` to run the online CPNav policy. Pass the calibrated
threshold from the offline stage with `--threshold`.

```bash
python scripts/run_online.py \
  --config config/CPnav_5_nano_131.yaml \
  --log_dir logs/online_single \
  --episode 100 \
  --threshold 0.059 \
  --mode cp_nav
```

Key arguments:

1. `--threshold`: Calibrated score threshold used by `cp_nav` mode, gathered from the offline calibration results after applying the conformal prediction bound.
2. `--mode`: Online method to run. Options are `cp_nav`, `simple_set`, `prompt_set`, and `ensemble_set`.
3. `--alpha`: Prediction-set threshold used by `simple_set` and `ensemble_set`.

To accelerate simulation, batch running is also available and is the recommended
way to run offline calibration or online evaluation across multiple episodes.

### Command 3: Batch Run Offline Calibration

Use `scripts/batch_run_offline.py` to run offline calibration on many randomly
sampled episodes in parallel.

```bash
python scripts/batch_run_offline.py \
  --config config/CPnav_5_nano_131.yaml \
  --batch_name offline_calibration \
  --n_exp 300 \
  --max_jobs 10 \
  --seed 42
```

The batch output is saved to `logs/<batch_name>/`. Each episode writes its own
run directory, and terminal logs are saved under `logs/<batch_name>/terminal/`.

### Command 4: Batch Run Online Implementation

Use `scripts/batch_run_online.py` to evaluate online navigation on many randomly
sampled episodes in parallel.

```bash
python scripts/batch_run_online.py \
  --config config/CPnav_5_nano_131.yaml \
  --batch_name online_cp_nav \
  --n_exp 100 \
  --max_jobs 15 \
  --seed 100 \
  --threshold 0.059 \
  --mode cp_nav
```

The online batch runner uses the same online arguments as `run_online.py`,
including `--threshold`, `--mode`, and `--alpha`.

## Method Summary

CPNAV uses a calibrated decision-making pipeline for object-goal navigation.

### Step 1: Action Proposal

Candidate motion primitives are generated from RGB-D observations while enforcing collision safety. The agent uses depth information to identify navigable regions and proposes feasible motion directions.

### Step 2: VLM Scoring

A vision-language model scores each candidate action based on semantic relevance to the target object. These scores estimate which actions are more likely to move the agent toward the goal.

### Step 3: Tree Search

The agent maintains a search tree during exploration. If a branch becomes unpromising or all admissible actions are exhausted, the agent performs local or global backtracking to continue searching from a previous state.

### Step 4: Conformal Prediction

Offline calibration produces a threshold that filters low-confidence actions during online deployment. This helps reduce overconfident but unreliable VLM decisions.

### Step 5: Deployment

During deployment, the calibrated safety set determines which actions are admissible for execution. The agent selects the highest-scoring admissible action and uses backtracking when no reliable action is available.

## Notes

1. Run all commands from the root `CPnav` directory unless stated otherwise.
2. Make sure `OPENAI_API_KEY` is set in the `.env` file before running VLM-based experiments.
3. Make sure the HM3D scene files and episode files match the expected `data/` directory structure.
4. Use `conda activate CPnav` before running scripts.
5. Do not commit `.env` or any private API keys to GitHub.

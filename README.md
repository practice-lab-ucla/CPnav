# CPNAV

**CPNAV** is a calibrated vision-language navigation framework for object-goal search in unseen indoor environments.

The method combines RGB-D based collision-aware action proposal, VLM-based semantic action scoring, tree-search exploration, and conformal prediction calibration. The conformal prediction layer filters low-confidence actions during navigation, improving reliability and path efficiency compared with uncalibrated VLM action selection.

## Overview

CPNAV is designed for map-free object-goal navigation. Given a target object category, such as `chair`, `bed`, or `toilet`, the agent explores an unseen indoor environment using RGB-D observations.

At each step, CPNAV:

1. Builds a local navigability map from depth input.
2. Generates collision-free candidate motion primitives.
3. Uses a vision-language model to score candidate actions.
4. Applies a calibrated conformal prediction threshold to filter unreliable actions.
5. Executes the best admissible action using a tree-search policy with backtracking.

## Installation

Create the conda environment and install all dependencies, including `habitat-sim`.

Make sure you are in the root `CPnav` directory:

```bash
conda env create -f environment.yml
conda activate CPnav
```



## OpenAI API Setup

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

Do not commit your `.env` file to GitHub. Make sure `.env` is listed in your `.gitignore` file:

```bash
.env
```

## Data Setup

This project uses a data setup similar to the one described in the [VLMnav repository](https://github.com/Jirl-upenn/VLMnav/blob/main/readme.md) for the paper [*End-to-End Navigation with Vision-Language Models: Transforming Spatial Reasoning into Question-Answering*](https://arxiv.org/pdf/2411.05755).

CPNAV uses HM3D scenes together with ObjectNav HM3D v2 episode files.

## Habitat-Matterport 3D Research Dataset

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

## Benchmark Datasets

Download the GOAT-Bench episode dataset from:

[Google Drive download link](https://drive.google.com/file/d/1N0UbpXK3v7oTphC4LoDqlNeMHbrwkbPe/view?usp=sharing)

For ObjectNav HM3D v2 episodes and HM3D scene assets, organize the downloaded data as follows:

```text
data/
├── datasets/
│   └── objectnav_hm3d_v2/
│       └── val/
│           ├── val.json.gz
│           └── content/
│               └── *.json.gz
│
└── versioned_data/
    └── hm3d-0.2/
        └── hm3d/
            ├── hm3d_annotated_basis.scene_dataset_config.json
            └── val/
                └── <scene_id>/
                    ├── <scene_id>.glb
                    ├── <scene_id>.navmesh
                    ├── <scene_id>.semantic.glb
                    └── <scene_id>.semantic.txt
```

## Running an Offline Experiment

Example command:

```bash
python scripts/run_offline.py \
  --config config/CPnav_5_nano_101.yaml \
  --log_dir logs/single_test \
  --episode 100
```

Arguments:

- `--config`: Path to the experiment configuration file.
- `--log_dir`: Directory where logs and results will be saved.
- `--episode`: Episode index to run.

## Project Structure

A typical project layout is:

```text
CPnav/
├── CPnav/
│   ├── runner.py
│   └── ...
├── config/
│   └── CPnav_5_nano_101.yaml
├── scripts/
│   └── run_offline.py
├── data/
│   ├── datasets/
│   └── versioned_data/
├── logs/
├── environment.yml
├── requirements.txt
└── README.md
```

## Method Summary

CPNAV uses a calibrated decision-making pipeline for object-goal navigation:

- **Action proposal:** Candidate motion primitives are generated from RGB-D observations while enforcing collision safety.
- **VLM scoring:** A vision-language model scores each action based on semantic relevance to the target object.
- **Tree search:** The agent maintains a search tree and performs local or global backtracking when needed.
- **Conformal prediction:** Offline calibration produces a threshold that filters low-confidence actions during online deployment.
- **Deployment:** The calibrated safety set determines which actions are admissible for execution.

## Notes

- Run all commands from the root `CPnav` directory unless stated otherwise.
- Make sure `OPENAI_API_KEY` is set before running VLM-based experiments.
- Make sure the HM3D scene files and episode files match the expected `data/` directory structure.
- Use `conda activate CPnav` before running scripts.
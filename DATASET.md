# Dataset Setup

This page covers downloading the scene and episode datasets used by CPNAV and
putting them where the configs expect them. Return to the
[README](README.md) when you are done.

CPNAV follows a data setup similar to the one used in the
[VLMnav repository](https://github.com/Jirl-upenn/VLMnav/blob/main/readme.md) for the paper
[*End-to-End Navigation with Vision-Language Models: Transforming Spatial Reasoning into Question-Answering*](https://arxiv.org/pdf/2411.05755).

- **HM3D** (default): Steps 1–2.
- **MP3D** (only for the MP3D experiments): Step 3.
- Step 4 shows the final `data/` layout.

Run all commands from the root `CPnav` directory. The configs read fixed paths
under `data/`, but the data itself can live anywhere: download it wherever you
like and symlink (or move) each folder to its required location.

## Step 1: Download the HM3D Scenes

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
  --uids hm3d_val_v0.2 \
  --data-path data/
```

This creates `data/versioned_data/hm3d-0.2/hm3d/`, which is already the
required location.

## Step 2: Download the HM3D ObjectNav Episodes

```bash
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip
unzip objectnav_hm3d_v2.zip -d data/datasets/
```

This creates `data/datasets/objectnav_hm3d_v2/` (with `train/`, `val/`, `val_mini/`).

## Step 3: Set Up MP3D (only for the MP3D experiments)

CPNAV uses the MP3D package from the [SG-Nav repo](https://github.com/bagh2178/SG-Nav)
(see its "Step 1 (Dataset)"): one zip (about 665 MB) with the 11 MP3D `val`
scenes and the ObjectNav MP3D v1 `val` episodes. MP3D is subject to the
[Matterport3D Terms of Use](https://niessner.github.io/Matterport/); agree to
them before using the data.

1. Download and unzip it anywhere you like (use an absolute path):

   ```bash
   MP3D_DIR=/path/to/your/mp3d_download
   mkdir -p "$MP3D_DIR"
   wget -O "$MP3D_DIR/MatterPort3D.zip" 'https://cloud.tsinghua.edu.cn/f/03e0ca1430a344efa72b/?dl=1'
   unzip "$MP3D_DIR/MatterPort3D.zip" -d "$MP3D_DIR"
   ```

   This gives:

   ```text
   MatterPort3D/
   ├── mp3d/                     # scenes: <scene_id>/<scene_id>.glb (11 scenes)
   └── objectnav/mp3d/v1/val/    # episodes: val.json.gz, content/<scene_id>.json.gz
   ```

2. Link the two folders to the locations the configs read. The download stays
   where it is:

   ```bash
   mkdir -p data/datasets/scene_datasets data/datasets/objectnav/mp3d
   ln -s "$MP3D_DIR/MatterPort3D/mp3d"              data/datasets/scene_datasets/mp3d
   ln -s "$MP3D_DIR/MatterPort3D/objectnav/mp3d/v1" data/datasets/objectnav/mp3d/v1
   ```

   Use `mv` instead of `ln -s` if you would rather move the folders.

## Step 4: Check the `data/` Layout

Once done, `data/` should look like this (any folder may be a symlink). These
are the paths the configs (`config/CPnav_5_nano_131*.yaml`) read from:

```text
data/
├── versioned_data/                          # HM3D scenes (Step 1)
│   └── hm3d-0.2/
│       └── hm3d/
│           ├── hm3d_annotated_basis.scene_dataset_config.json
│           └── val/
│               └── 00800-TEEsavR23oF/       # <number>-<scene_id>
│                   ├── TEEsavR23oF.basis.glb
│                   ├── TEEsavR23oF.basis.navmesh
│                   ├── TEEsavR23oF.semantic.glb
│                   └── TEEsavR23oF.semantic.txt
│
└── datasets/
    ├── objectnav_hm3d_v2/                    # HM3D ObjectNav episodes (Step 2)
    │   └── val/
    │       ├── val.json.gz
    │       └── content/<scene_id>.json.gz
    │
    ├── scene_datasets/
    │   └── mp3d/                             # MP3D scenes (Step 3)
    │       └── 2azQ1b91cZZ/                  # <scene_id>
    │           └── 2azQ1b91cZZ.glb
    │
    └── objectnav/
        └── mp3d/
            └── v1/                           # MP3D ObjectNav episodes (Step 3)
                └── val/
                    ├── val.json.gz
                    └── content/<scene_id>.json.gz
```

Quick check for MP3D (both should print `11`):

```bash
ls data/datasets/scene_datasets/mp3d | wc -l
ls data/datasets/objectnav/mp3d/v1/val/content | wc -l
```

Paths per dataset (from the configs):

| dataset | scenes                              | episodes                          | config                             |
| ------- | ----------------------------------- | --------------------------------- | ---------------------------------- |
| HM3D    | `data/versioned_data/hm3d-0.2/hm3d` | `data/datasets/objectnav_hm3d_v2` | `config/CPnav_5_nano_131.yaml`     |
| MP3D    | `data/datasets/scene_datasets/mp3d` | `data/datasets/objectnav/mp3d/v1` | `config/CPnav_5_nano_131_mp3d.yaml`|

HM3D uses its bundled `.basis.navmesh`. The MP3D scenes come with only the
`.glb`, so the simulator computes the navmesh from the scene geometry when the
scene loads.

To switch a config between datasets, set the dataset name (`run.sh` picks the
right config for you, so you usually don't touch this):

```yaml
env_cfg:
  dataset:
    name: hm3d  # or mp3d
```

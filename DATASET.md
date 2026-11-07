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

MP3D needs two folders. Download them anywhere, then link each one to its
required location:

| what                    | required location                    | must contain                                    |
| ----------------------- | ------------------------------------ | ----------------------------------------------- |
| MP3D scenes             | `data/datasets/scene_datasets/mp3d/` | `<scene_id>/<scene_id>.glb`                     |
| ObjectNav MP3D episodes | `data/datasets/objectnav/mp3d/v1/`   | `val/val.json.gz`, `val/content/<scene_id>.json.gz` |

Pick a download folder (any location, absolute path):

```bash
MP3D_DIR=/path/to/your/mp3d_download
mkdir -p "$MP3D_DIR"
```

### 3a: MP3D scenes

1. Request access by signing the Matterport3D Terms of Use
   (<https://niessner.github.io/Matterport/>). Matterport replies with the
   download script `download_mp.py`.
2. Download the Habitat version of the scenes (`mp3d_habitat.zip`). The script
   from Matterport is written for Python 2, so run it with `python2`:

   ```bash
   python2 download_mp.py --task_data habitat -o "$MP3D_DIR"
   ```

   This saves `$MP3D_DIR/v1/tasks/mp3d_habitat.zip`. The script then asks
   whether to continue with the full raw dataset; press `Ctrl-C` to stop
   there, it is not needed.
3. Unzip it. This gives a `mp3d/` folder with one subfolder per scene:

   ```bash
   unzip "$MP3D_DIR/v1/tasks/mp3d_habitat.zip" -d "$MP3D_DIR"
   ```

4. Link the `mp3d/` folder to its required location:

   ```bash
   mkdir -p data/datasets/scene_datasets
   ln -s "$MP3D_DIR/mp3d" data/datasets/scene_datasets/mp3d
   ```

Only the 11 MP3D `val` scenes are used, so the other scene folders can be
deleted to save space: `2azQ1b91cZZ`, `8194nk5LbLH`, `EU6Fwq7SyZv`,
`oLBMNvg9in8`, `pLe4wQe7qrG`, `QUCTc6BB5sX`, `TbHJrupSAjP`, `X7HyMhZNoso`,
`x8F5xyUWy9e`, `Z6MFQCViBuw`, `zsNo4HB9uLZ`.

CPNAV reads only `<scene_id>.glb` from each scene folder; the other files in
the archive (`.house`, `_semantic.ply`) are not used.

### 3b: ObjectNav MP3D episodes

```bash
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip
unzip objectnav_mp3d_v1.zip -d "$MP3D_DIR/objectnav_mp3d_v1"
mkdir -p data/datasets/objectnav/mp3d
ln -s "$MP3D_DIR/objectnav_mp3d_v1" data/datasets/objectnav/mp3d/v1
```

Notes:

- `m3d` in the download URL is correct (it is the official Habitat path).
- This zip has no top-level folder (it unpacks straight into `train/`, `val/`,
  `val_mini/`), so always pass `-d` to `unzip`.

Alternatively, the [SG-Nav repo](https://github.com/bagh2178/SG-Nav) documents
an MP3D + ObjectNav data setup; link its scene and episode folders to the same
two locations.

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
    │   └── mp3d/                             # MP3D scenes (Step 3a)
    │       └── 2azQ1b91cZZ/                  # <scene_id>
    │           └── 2azQ1b91cZZ.glb
    │
    └── objectnav/
        └── mp3d/
            └── v1/                           # MP3D ObjectNav episodes (Step 3b)
                └── val/
                    ├── val.json.gz
                    └── content/<scene_id>.json.gz
```

Quick check for MP3D (both should print `11`; the first may be higher if you
kept all scenes):

```bash
ls data/datasets/scene_datasets/mp3d | wc -l
ls data/datasets/objectnav/mp3d/v1/val/content | wc -l
```

Paths per dataset (from the configs):

| dataset | scenes                              | episodes                          | config                             |
| ------- | ----------------------------------- | --------------------------------- | ---------------------------------- |
| HM3D    | `data/versioned_data/hm3d-0.2/hm3d` | `data/datasets/objectnav_hm3d_v2` | `config/CPnav_5_nano_131.yaml`     |
| MP3D    | `data/datasets/scene_datasets/mp3d` | `data/datasets/objectnav/mp3d/v1` | `config/CPnav_5_nano_131_mp3d.yaml`|

HM3D uses its bundled `.basis.navmesh`. For MP3D, if a `<scene_id>.navmesh` sits
next to the `.glb`, CPNAV loads it; otherwise the simulator computes the navmesh
from the scene geometry when the scene loads.

To switch a config between datasets, set the dataset name (`run.sh` picks the
right config for you, so you usually don't touch this):

```yaml
env_cfg:
  dataset:
    name: hm3d  # or mp3d
```

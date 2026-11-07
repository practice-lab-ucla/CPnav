# Dataset Setup

This page covers downloading the scene and episode datasets used by CPNAV and
placing them where the configs expect them. Return to the
[README](README.md) when you are done.

CPNAV follows a data setup similar to the one used in the
[VLMnav repository](https://github.com/Jirl-upenn/VLMnav/blob/main/readme.md) for the paper
[*End-to-End Navigation with Vision-Language Models: Transforming Spatial Reasoning into Question-Answering*](https://arxiv.org/pdf/2411.05755).

By default, CPNAV uses HM3D scenes with ObjectNav HM3D v2 episodes (Steps 1–2).
The MP3D (Matterport3D) experiments additionally need the MP3D scenes and
ObjectNav MP3D episodes (Step 3). Step 4 shows where everything goes.

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
  --uids hm3d_val_v0.2
```

## Step 2: Download the HM3D ObjectNav Episodes

Download the ObjectNav HM3D v2 episode dataset from:

[https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip)

## Step 3: Download MP3D (only for the MP3D experiments)

**Option A — packaged with SG-Nav (simplest).** The
[SG-Nav repo](https://github.com/bagh2178/SG-Nav) documents an MP3D + ObjectNav
data setup; follow its data instructions and place the scenes under
`data/datasets/scene_datasets/mp3d/` and the episodes under
`data/datasets/objectnav/mp3d/v1/` as shown in Step 4.

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

## Step 4: Organize the `data/` Directory

Place everything under `data/` in the root `CPnav` directory exactly as below.
These are the paths the configs (`config/CPnav_5_nano_131*.yaml`) read from:

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

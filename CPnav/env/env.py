import gzip
import json
import logging
import os

import numpy as np

from scipy.spatial.transform import Rotation as R
from pathlib import Path
from typing import Union

from .sim_wrapper import SimWrapper, PolarAction, TeleopAction, NULL_ACTION
from ..utils import *


REPO_ROOT = Path(__file__).resolve().parents[2]


def get_quat_from_heading_angle(heading_angle_deg: float) -> np.ndarray:
    """
    Convert a heading angle in degrees (rotation around Y-axis) into a quaternion.

    Args:
        heading_angle_deg: Heading angle in degrees.

    Returns:
        np.ndarray: Quaternion in [x, y, z, w] format.
    """
    angle_rad = np.radians(heading_angle_deg)
    rot = R.from_euler("y", angle_rad)
    return rot.as_quat()  # [x, y, z, w]


class ObjectNavEnv:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.agent_cfg = self.cfg["agent_cfg"]
        self.sim_cfg = self.cfg["sim_cfg"]
        self.env_cfg = self.cfg["env_cfg"]

        # Make sim paths repo-root-relative
        self.sim_cfg["scene_path_root"] = str((REPO_ROOT / self.sim_cfg["scene_path_root"]).resolve())
        self.sim_cfg["scene_config"] = str((REPO_ROOT / self.sim_cfg["scene_config"]).resolve())

        # Make dataset path repo-root-relative (optional, you already do it later)
        self.env_cfg["dataset_root_path"] = str((REPO_ROOT / self.env_cfg["dataset_root_path"]).resolve())

        self._initialize_experiment()

    def _initialize_experiment(self) -> None:
        self.all_episodes = []
        self.goals = {}

        # Load all episodes + goals for the chosen split
        split = self.env_cfg["split"]
        dataset_root_path = os.path.join(REPO_ROOT, self.env_cfg['dataset_root_path'])
        dataset_content_path = os.path.join(dataset_root_path, split, "content")

        for f in sorted(os.listdir(dataset_content_path)):
            file_path = os.path.join(dataset_content_path, f)
            with gzip.open(file_path, "rt") as gz:
                js = json.load(gz)
                hsh = f.split(".")[0]
                # goals_by_category stored under this key
                self.goals[hsh] = js["goals_by_category"]
                self.all_episodes += js["episodes"]

        # self.all_episodes = sorted(self.all_episodes, key=lambda e: int(e["episode_id"]))

        self.num_episodes = len(self.all_episodes)
        logging.info(
            f"Initialized ObjectNavEnv with {self.num_episodes} episodes (split={split})."
        )

    def initialize_episode(self, episode_ndx: int):
        episode = self.all_episodes[episode_ndx]
        # Example episode["scene_id"]:
        #   "hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb"
        scene_parts = episode["scene_id"].split("/")[1:]  # drop leading root
        # scene_parts = [ "val", "<scene_folder>", "<scene_glb>" ]

        # Scene ID (used by SimWrapper)
        self.sim_cfg["scene_id"] = scene_parts[1][2:5]

        # Full path to the .glb file
        self.sim_cfg["scene_path"] = os.path.join(
            self.sim_cfg["scene_path_root"], 
            self.env_cfg['split'], 
            scene_parts[1], 
            scene_parts[2]
        )

        self.simWrapper = SimWrapper(self.sim_cfg)

        goals = self.goals[scene_parts[1][6:]]
        all_objects = goals[f"{scene_parts[-1]}_{episode['object_category']}"]
        episode_goal_category = episode["object_category"]

        logging.info(
            f"RUNNING EPISODE {episode_ndx} with target {episode_goal_category}"
            f"{episode['object_category']} ({len(all_objects)} instances). "
        )

        # Initial agent pose
        self.init_pos = np.array(episode["start_position"])
        self.init_rot = np.array(episode["start_rotation"])

        logging.info(
            f"Initial agent position: {self.init_pos}, rotation (quat): {self.init_rot}"
        )

        # Initialize simulator state and get first observation
        self.simWrapper.set_state(pos=self.init_pos, quat=self.init_rot)
        self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"
        self.step_count = 0

        return episode_goal_category

    def step(self, agent_action: Union[PolarAction, TeleopAction]):
        self.step_count += 1
        if isinstance(agent_action, PolarAction):
            agent_action.r = agent_action.r * self.agent_cfg["discourage_ratio"]
            # agent_action.theta = agent_action.theta * (-1)  # TODO
            return self.simWrapper.step(agent_action)
        elif isinstance(agent_action, TeleopAction):
            return self.simWrapper.teleport_to_xz(agent_action)
    
    def reset(self):
        self.step_count = 0
        return self.simWrapper.step(NULL_ACTION)
    
    def close(self):
        self.simWrapper.close()

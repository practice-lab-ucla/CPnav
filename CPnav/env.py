import gzip
import json
import logging
import math
import os
import random

import habitat_sim
import numpy as np

from scipy.spatial.transform import Rotation as R

from sim_wrapper import PolarAction, SimWrapper
from utils import *  # expects log_exception, etc.


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

    def _initialize_experiment(self) -> None:
        self.all_episodes = []
        self.goals = {}

        # Load all episodes + goals for the chosen split
        split = self.cfg["split"]
        dataset_root = os.path.join(self.cfg['root_path'], split, "content")

        for f in sorted(os.listdir(dataset_root)):
            file_path = os.path.join(dataset_root, f)
            with gzip.open(file_path, "rt") as gz:
                js = json.load(gz)
                hsh = f.split(".")[0]
                # goals_by_category stored under this key
                self.goals[hsh] = js["goals_by_category"]
                self.all_episodes += js["episodes"]

        self.num_episodes = len(self.all_episodes)
        logging.info(
            f"Initialized ObjectNavEnv with {self.num_episodes} episodes (split={split})."
        )

    def _initialize_episode(self, episode_ndx: int):
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
            self.cfg['split'], 
            scene_parts[1], 
            scene_parts[2]
        )

        # Create simulator wrapper for this scene
        self.simWrapper = SimWrapper(self.sim_cfg)

        # Lookup all goal objects and their view points
        # Use the part of the scene folder name after the first 6 chars to
        # index into self.goals (matches dataset naming convention).
        goals = self.goals[scene_parts[1][6:]]
        all_objects = goals[f"{scene_parts[-1]}_{episode['object_category']}"]

        logging.info(
            f"RUNNING EPISODE {episode_ndx} with target "
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

    def step(self, agent_actions):
        self.step_count += 1
        return self.simWrapper.step(agent_actions)


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


class Env:
    """
    Base class for creating an environment for embodied navigation tasks.

    Subclasses must implement:
      - _initialize_experiment(self)
      - _initialize_episode(self, episode_ndx)
      - _step_env(self, obs, episode_ndx)
    """

    def __init__(self, cfg: dict):
        """
        Initialize the environment with the provided configuration.

        Args:
            cfg: Configuration dictionary containing environment, simulation,
                 and agent settings. Expected keys:
                    - "task": str
                    - "env_cfg": dict
                    - "sim_cfg": dict
                    - "agent_cfg": dict (inside cfg passed to _initialize_agent)
        """
        # Distance to goal (kept for compatibility; not used in this minimal version)
        self.distance_to_goal = float("inf")

        # Store config sections
        self.cfg = cfg["env_cfg"]
        self.sim_cfg = cfg["sim_cfg"]

        # Simulation wrapper (set by subclasses in _initialize_episode)
        self.simWrapper: SimWrapper | None = None

        # Episode-related attributes
        self.num_episodes: int = 0
        # self.agent_distance_traveled: float = 0.0
        # self.prev_agent_position: np.ndarray | None = None

        # Subclass-specific experiment setup
        self._initialize_experiment()


    def _initialize_agent(self, cfg: dict) -> None:
        """
        Initialize the agent.

        The agent class is taken from cfg["agent_cls"] and instantiated with
        cfg["agent_cfg"]. The sensor configuration is taken from sim_cfg.
        """

        cfg["agent_cfg"]["sensor_cfg"] = cfg["sim_cfg"]["sensor_cfg"]

    # -------------------------------------------------------------------------
    # Abstract / task-specific hooks
    # -------------------------------------------------------------------------

    def _initialize_experiment(self) -> None:
        """
        Set up the experiment and initialize any required variables.

        Must set:
          - self.num_episodes
          - any dataset / scene configuration used by the environment.

        Must be implemented in derived classes.
        """
        raise NotImplementedError

 
    def _initialize_episode(self):
        """
        Reset bookkeeping for a new episode.

        Subclasses must call this via super()._initialize_episode(episode_ndx)
        and then perform their own initialization (including creating the
        SimWrapper and returning the first observation).

        Args:
            episode_ndx: Index of the episode to initialize.
        """
        self.step_count = 0
        self.init_pos = None
        self.init_rot = None
        # self.agent_distance_traveled = 0.0
        # self.prev_agent_position = None

    def _step_env(self, action):
        """
        Base environment step. Subclasses usually override this.
        """
        raise NotImplementedError
        

    def _post_episode(self) -> None:
        """
        Called after the episode is complete.

        Resets simulation and agent, and logs episode completion.
        """
        print("Episode finished, finalizing logs and metrics...")

        # Reset simulation and agent between episodes
        if self.simWrapper is not None:
            self.simWrapper.reset()
        self.agent.reset()

        logging.info("\n=================== RUN COMPLETE ===================\n")


# =============================================================================
# ObjectNav Environment
# =============================================================================


class ObjectNavEnv(Env):
    """
    Environment for the ObjectNav task, extending the base Env class.

    Responsible for:
      - Loading ObjectNav episodes.
      - Configuring scenes.
      - Running episodes with an ObjectNav agent.
    """

    def _initialize_experiment(self) -> None:
        """
        Initialize the experiment by setting up the dataset split, scene
        configuration, and goals.
        """
        self.all_episodes = []
        self.goals = {}

        # Set the HM3D scene dataset config
        self.sim_cfg[
            "scene_config"
        ] = "../data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

        # Load all episodes + goals for the chosen split
        split = self.cfg["split"]
        dataset_root = f"{self.cfg['root_path']}/{split}/content"

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
        """
        Initialize one ObjectNav episode.

        This sets up:
          - The scene path and simulator.
          - The target object category and its positions.
          - The initial agent position.

        Args:
            episode_ndx: The index of the episode to initialize.

        Returns:
            dict: Initial observation from the simulator.
        """
        super()._initialize_episode()

        episode = self.all_episodes[episode_ndx]

        # Example episode["scene_id"]:
        #   "hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb"
        scene_parts = episode["scene_id"].split("/")[1:]  # drop leading root
        # scene_parts = [ "val", "<scene_folder>", "<scene_glb>" ]

        # Scene ID (used by SimWrapper)
        self.sim_cfg["scene_id"] = scene_parts[1][2:5]

        # Full path to the .glb file
        self.sim_cfg["scene_path"] = (
            f"../data/scene_datasets/hm3d/"
            f"{self.cfg['split']}/{scene_parts[1]}/{scene_parts[2]}"
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

        # First "null" step to obtain initial observation
        obs = self.simWrapper.step(PolarAction.null)
        return obs

    def step(self, agent_actions):
        """
        Single environment step for ObjectNav.

        Pipeline:
          - Attach goal + episode info into obs.
          - Let the agent produce an action.
          - Return that action to be executed by the simulator.

        Args:
            obs: Current observation dictionary.
            episode_ndx: Index of the current episode.

        Returns:
            PolarAction | None: Action for the simulator, or PolarAction.stop
                                 if the episode is done.
        """
        self.step_count += 1
        return self.simWrapper.step(agent_actions)

    def reset(self):
        """
        Single environment step for ObjectNav.

        Pipeline:
          - Attach goal + episode info into obs.
          - Let the agent produce an action.
          - Return that action to be executed by the simulator.

        Args:
            obs: Current observation dictionary.
            episode_ndx: Index of the current episode.

        Returns:
            PolarAction | None: Action for the simulator, or PolarAction.stop
                                 if the episode is done.
        """
        obs = self.simWrapper.step(PolarAction.null)
        self.simWrapper.reset()
        return obs



import gzip
import json
import logging
import os
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Tuple
from scipy.spatial.transform import Rotation as R

# Assuming these imports exist in your project structure
from .sim_wrapper import SimWrapper, PolarAction, TeleopAction, NULL_ACTION

REPO_ROOT = Path(__file__).resolve().parents[2]

def get_quat_from_heading_angle(heading_angle_deg: float) -> np.ndarray:
    """
    Convert a heading angle in degrees (rotation around Y-axis) into a quaternion.
    """
    angle_rad = np.radians(heading_angle_deg)
    rot = R.from_euler("y", angle_rad)
    return rot.as_quat()  # [x, y, z, w]

class ObjectNavEnv:
    def __init__(self, cfg: Dict[str, Any]):
        """
        Args:
            cfg: The dictionary loaded from the clean YAML file.
        """
        self.cfg = cfg
        
        # 1. Unpack the clean config sections
        self.env_cfg = self.cfg["env_cfg"]
        self.sim_cfg = self.cfg["sim_cfg"]
        self.agent_cfg = self.cfg["agent_cfg"]

        # 2. Resolve Paths (Make them absolute based on REPO_ROOT)
        self._resolve_paths()

        # 3. Load Dataset Index
        self.goals = {}
        self.all_episodes = []
        self._initialize_experiment()

        # State trackers
        self.simWrapper = None
        self.init_pos = None
        self.init_rot = None
        self.curr_run_name = None
        self.step_count = 0

    def _resolve_paths(self):
        """Resolves relative paths in config to absolute paths."""
        # Sim paths
        self.sim_cfg["scene_path_root"] = str((REPO_ROOT / self.sim_cfg["scene_path_root"]).resolve())
        self.sim_cfg["scene_config"] = str((REPO_ROOT / self.sim_cfg["scene_config"]).resolve())

        # Dataset path
        self.env_cfg["dataset_root_path"] = str((REPO_ROOT / self.env_cfg["dataset_root_path"]).resolve())

    def _initialize_experiment(self) -> None:
        """Loads all episode JSONs from the content folder."""
        split = self.env_cfg["split"]
        dataset_root = Path(self.env_cfg['dataset_root_path'])
        content_path = dataset_root / split / "content"

        if not content_path.exists():
            raise FileNotFoundError(f"Dataset content not found at: {content_path}")

        logging.info(f"Loading episodes from {content_path}...")

        # Iterate over compressed JSONs
        for file_path in sorted(content_path.glob("*.json.gz")):
            with gzip.open(file_path, "rt") as gz:
                data = json.load(gz)
                
                # The filename (minus extensions) is usually the scene hash
                scene_hash = file_path.name.split(".")[0]
                
                # Store goals and episodes
                self.goals[scene_hash] = data["goals_by_category"]
                self.all_episodes.extend(data["episodes"])

        self.num_episodes = len(self.all_episodes)
        logging.info(
            f"Initialized ObjectNavEnv with {self.num_episodes} episodes (split={split})."
        )

    def initialize_episode(self, episode_ndx: int) -> str:
        """
        Sets up the simulator for the specific episode index.
        Returns: The goal category string.
        """
        episode = self.all_episodes[episode_ndx]
        
        # Parse Scene ID from HM3D format: "hm3d/val/HASH/HASH.basis.glb"
        # We split by '/' to get parts.
        raw_scene_id = episode["scene_id"]
        scene_parts = raw_scene_id.split("/") 
        
        # Expecting parts: ['hm3d', 'val', 'HASH', 'HASH.basis.glb']
        # Or relative: ['val', 'HASH', 'HASH.basis.glb']
        # We grab the hash folder and the filename.
        scene_hash = scene_parts[-2]  # The folder name (e.g., 00877-4ok3usBNeis)
        glb_filename = scene_parts[-1] # The file name

        # 1. Configure Sim for this Scene
        # Note: scene_id in sim_cfg is often used for logging or display
        self.sim_cfg["scene_id"] = scene_hash 

        # Construct full path to the GLB
        self.sim_cfg["scene_path"] = os.path.join(
            self.sim_cfg["scene_path_root"], 
            self.env_cfg['split'], 
            scene_hash, 
            glb_filename
        )

        # 2. Re-initialize Simulator
        if self.simWrapper:
            self.simWrapper.close()
        self.simWrapper = SimWrapper(self.sim_cfg)

        # 3. Retrieve Goals
        # HM3D hashes often have a prefix (e.g., "00877-") that is stripped for the goal lookup key
        # We assume the key is the part after the first 6 chars if the format holds, 
        # or we try to derive it. Based on your old code: `scene_parts[1][6:]`
        # Let's try to be robust:
        short_hash = scene_hash.split('-')[-1] # "4ok3usBNeis" from "00877-4ok3usBNeis"
        
        target_category = episode["object_category"]
        
        # Verify we have goals for this scene
        if short_hash not in self.goals:
            # Fallback: try the full hash if short failed, or log warning
            logging.warning(f"Goal hash {short_hash} not found. Trying {scene_hash}...")
            if scene_hash in self.goals:
                short_hash = scene_hash
            else:
                logging.error(f"Could not find goals for scene {scene_hash}")

        # 4. Set Agent State
        self.init_pos = np.array(episode["start_position"])
        self.init_rot = np.array(episode["start_rotation"])

        logging.info(
            f"EPISODE {episode_ndx} | Scene: {scene_hash} | Target: {target_category} | "
            f"Start: {self.init_pos}"
        )

        self.simWrapper.set_state(pos=self.init_pos, quat=self.init_rot)
        self.curr_run_name = f"{episode_ndx}_{short_hash}"
        self.step_count = 0

        return target_category

    def step(self, agent_action: Union[PolarAction, TeleopAction]):
        """
        Executes an action in the simulator.
        Handles Polar movement (Forward/Turn) and Teleportation.
        """
        self.step_count += 1
        
        if isinstance(agent_action, PolarAction):
            # Apply scaling from agent configuration
            # New config location: agent_cfg -> discourage_ratio
            ratio = self.agent_cfg.get("discourage_ratio", 1.0)
            
            # Create a copy or modify in place (modifying in place is faster)
            scaled_action = PolarAction(
                r = agent_action.r * ratio,
                theta = agent_action.theta # TODO: Verify if inversion is needed
            )
            return self.simWrapper.step(scaled_action)
            
        elif isinstance(agent_action, TeleopAction):
            # TeleportAction is usually just an (x, z) tuple or class wrapping it
            return self.simWrapper.teleport_to_xz(agent_action)
        
        else:
            raise ValueError(f"Unknown action type: {type(agent_action)}")
    
    def reset(self):
        """Resets the agent to the current episode's start state via a NULL step."""
        self.step_count = 0
        return self.simWrapper.step(NULL_ACTION)
    
    def close(self):
        if self.simWrapper:
            self.simWrapper.close()
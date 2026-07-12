from __future__ import annotations

import glob
import gzip
import json
import logging

# Make Simulator dependencies optional!
try:
    import habitat_sim
except ImportError:
    habitat_sim = None

import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Tuple
from scipy.spatial.transform import Rotation as R

# Assuming these imports exist in your project structure
from .sim_wrapper import SimWrapper
from .actions import PolarAction, TeleopAction, STOP_ACTION, NULL_ACTION

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
        """Resolve dataset-specific paths and preserve legacy config keys."""
        self.dataset_cfg = self._resolve_dataset_cfg()

        for key in ("episodes_root", "scene_root", "scene_dataset_config"):
            value = self.dataset_cfg.get(key)
            if value:
                self.dataset_cfg[key] = str((REPO_ROOT / value).resolve())

        # Keep the previous key names populated so older code/logs still work.
        self.env_cfg["split"] = self.dataset_cfg["split"]
        self.env_cfg["dataset_root_path"] = self.dataset_cfg["episodes_root"]
        self.sim_cfg["scene_path_root"] = self.dataset_cfg["scene_root"]
        self.sim_cfg["scene_config"] = self.dataset_cfg.get("scene_dataset_config") or ""

    def _resolve_dataset_cfg(self) -> Dict[str, Any]:
        """Flatten the active dataset block, matching fai-nav's hm3d/mp3d switch."""
        ds = self.env_cfg.get("dataset")
        if isinstance(ds, dict):
            name = (ds.get("name") or "hm3d").lower()
            active = ds.get(name)
            if not isinstance(active, dict):
                if "episodes_root" in ds and "scene_root" in ds:
                    merged = dict(ds)
                    merged["name"] = name
                    merged.setdefault("split", self.env_cfg.get("split", "val"))
                    return merged
                raise KeyError(
                    f"dataset.name='{name}' but no env_cfg.dataset.{name} block found."
                )

            merged = {
                k: v for k, v in ds.items()
                if k not in ("hm3d", "mp3d")
            }
            merged.update(active)
            merged["name"] = name
            merged.setdefault("split", self.env_cfg.get("split", "val"))
            return merged

        # Legacy CPnav config shape: HM3D paths split between env_cfg and sim_cfg.
        return {
            "name": "hm3d",
            "split": self.env_cfg.get("split", "val"),
            "episodes_root": self.env_cfg["dataset_root_path"],
            "scene_root": self.sim_cfg["scene_path_root"],
            "scene_dataset_config": self.sim_cfg.get("scene_config"),
        }

    def _initialize_experiment(self) -> None:
        """Loads all episode JSONs from the content folder."""
        split = self.dataset_cfg["split"]
        dataset_root = Path(self.dataset_cfg["episodes_root"])
        split_path = dataset_root / split
        files = sorted((split_path / "content").glob("*.json.gz"))
        if not files:
            files = sorted(split_path.glob("*.json.gz"))
        if not files:
            raise FileNotFoundError(f"Dataset episodes not found under: {split_path}")

        logging.info(f"Loading episodes from {split_path}...")

        for file_path in files:
            with gzip.open(file_path, "rt") as gz:
                data = json.load(gz)

            # The content filename is usually the short scene id. Aggregate split
            # files still work because goal lookup is only used for run naming.
            scene_hash = file_path.name.removesuffix(".json.gz")
            self.goals[scene_hash] = data.get("goals_by_category", {})
            self.all_episodes.extend(data.get("episodes", []))

        self.num_episodes = len(self.all_episodes)
        if self.num_episodes == 0:
            raise RuntimeError(f"No episodes found under: {split_path}")
        logging.info(
            f"Initialized ObjectNavEnv with {self.num_episodes} episodes "
            f"(dataset={self.dataset_cfg['name']}, split={split})."
        )

    def _scene_glb(self, raw_scene_id: str) -> str:
        """Resolve Habitat episode scene_id to an absolute .glb path."""
        scene_root = Path(self.dataset_cfg["scene_root"])
        split = self.dataset_cfg["split"]
        parts = raw_scene_id.split("/")
        scene_folder, glb_name = parts[-2], parts[-1]

        # HM3D: scene_root/<split>/<scene_folder>/<scene>.basis.glb
        candidate = scene_root / split / scene_folder / glb_name
        if candidate.exists():
            return str(candidate)

        # MP3D: scene_root/<scene_folder>/<scene>.glb
        flat = scene_root / scene_folder / glb_name
        if flat.exists():
            return str(flat)

        hits = glob.glob(str(scene_root / "**" / glb_name), recursive=True)
        if hits:
            return hits[0]

        raise FileNotFoundError(f"Could not locate scene glb for {raw_scene_id}")

    def initialize_episode(self, episode_ndx: int) -> str:
        """
        Sets up the simulator for the specific episode index.
        Returns: The goal category string.
        """
        episode = self.all_episodes[episode_ndx]
        
        # Parse scene ids such as:
        # HM3D: hm3d_v0.2/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb
        # MP3D: mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb
        raw_scene_id = episode["scene_id"]
        scene_parts = raw_scene_id.split("/") 
        scene_hash = scene_parts[-2]  # The folder name (e.g., 00877-4ok3usBNeis)

        # 1. Configure Sim for this Scene
        # Note: scene_id in sim_cfg is often used for logging or display
        self.sim_cfg["scene_id"] = scene_hash 

        self.sim_cfg["scene_path"] = self._scene_glb(raw_scene_id)

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

    def start_to_goal_geodesic_distance(self, start_xz: Tuple[float], end_xz: Tuple[float]):
        pf = self.simWrapper.sim.pathfinder

        # --- Compute full shortest path from start → current ---
        shortest_path = habitat_sim.ShortestPath()
        shortest_path.requested_start = np.array([start_xz[0], 0, start_xz[1]])
        shortest_path.requested_end = np.array([end_xz[0], 0, end_xz[1]])

        found = pf.find_path(shortest_path)

        if found:
            dist_start_to_curr = float(shortest_path.geodesic_distance)
        else:
            dist_start_to_curr = float('inf')

        return dist_start_to_curr

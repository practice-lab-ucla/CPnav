import os
import cv2
import json
import yaml
from datetime import datetime
from typing import List, Tuple, Optional


def quaternion_to_wxyz(rotation):
    return [rotation.w, rotation.x, rotation.y, rotation.z]


class Logger:
    def __init__(self, episode_id, root_dir: str = "./logs"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(root_dir, f"episode_{episode_id}_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.step = 0

        # subfolders
        self.txt_dir = os.path.join(self.run_dir, "texts")
        self.json_dir = os.path.join(self.run_dir, "meta")
        self.img_dir = os.path.join(self.run_dir, "images")
        self.offline_dir = os.path.join(self.run_dir, "offline")
        self.terminal_dir = os.path.join(self.run_dir, "terminal")

        for d in [self.img_dir, self.txt_dir, self.json_dir, self.terminal_dir, self.offline_dir]:
            os.makedirs(d, exist_ok=True)

    def log_step(
        self,
        overlay_rgb,
        voxel_rgb,
        position,
        rotation,
        stop_text: str,
        act_text: str,
        actions: List[Tuple[float, float]],
        action_idx: Optional[int],
        confidence_scores: Optional[list],
        done: int,
        global_semantic_score: Optional[float],
    ):
        step_id = f"{self.step:04d}"

        # ---------- image ----------
        img_path = os.path.join(self.img_dir, f"{step_id}_overlay.png")
        voxel_path = os.path.join(self.img_dir, f"{step_id}_voxel.png")
        # OpenCV expects BGR
        cv2.imwrite(img_path, overlay_rgb[:, :, ::-1])
        cv2.imwrite(voxel_path, voxel_rgb[:, :, ::-1])

        # ---------- texts ----------
        with open(os.path.join(self.txt_dir, f"{step_id}_stop.txt"), "w") as f:
            f.write(stop_text)

        with open(os.path.join(self.txt_dir, f"{step_id}_action.txt"), "w") as f:
            f.write(act_text)

        # ---------- structured metadata ----------
        meta = {
            "step": self.step,
            "position":[float(p) for p in position],
            "rotation": quaternion_to_wxyz(rotation),
            "done": int(done),
            "global_semantic_score": global_semantic_score,
            "chosen_action_index": action_idx,
            "actions": [
                {"index": i+1, "r": float(r), "theta": float(theta)}
                for i, (r, theta) in enumerate(actions)
            ],
            "confidence_scores": confidence_scores,
        }

        with open(os.path.join(self.json_dir, f"{step_id}.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self.step += 1

    def save_terminal_data(
        self,
        goal_text: str,
        step_count: int,
        traversed_distance: float,
        start_xz: Tuple[float, float],
        end_xz: Tuple[float, float],
        geodesic_distance: float,
        history: List[str],
        last_image,
        threshold: float
    ):
        """
        Saves final episode metadata and the terminal image.
        """
        # 1. Save Terminal Image
        if last_image is not None:
            term_img_path = os.path.join(self.terminal_dir, f"{goal_text}.png")
            # Convert RGB to BGR for OpenCV
            cv2.imwrite(term_img_path, last_image[:, :, ::-1])

        # 2. Save Metadata JSON
        data = {
            "goal": goal_text,
            "step_count": int(step_count),
            "total_distance_traversed": float(traversed_distance),
            "start_position_xz": [float(start_xz[0]), float(start_xz[1])],
            "end_position_xz": [float(end_xz[0]), float(end_xz[1])],
            "geodesic_distance": float(geodesic_distance),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "history": [str(h) for h in history],
            "threshold": float(threshold)
        }

        meta_path = os.path.join(self.terminal_dir, "terminal_summary.json")
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=4)

    def save_failed_terminal_data(
        self,
        goal_text: str,
        step_count: int,
        traversed_distance: float,
        start_xz: Tuple[float, float],
        history: List[str],
        threshold: float
    ):
        """
        Saves final episode metadata and the terminal image.
        """

        # 2. Save Metadata JSON
        data = {
            "goal": goal_text,
            "step_count": int(step_count),
            "total_distance_traversed": float(traversed_distance),
            "start_position_xz": [float(start_xz[0]), float(start_xz[1])],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "history": [str(h) for h in history],
            "threshold": float(threshold)
        }

        meta_path = os.path.join(self.terminal_dir, "failure_summary.json")
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=4)

    def save_offline_data(
        self,
        score: float,
        best_path: List[Tuple[int]]
    ):
        """
        Saves final episode metadata and the terminal image.
        """

        data = {
            "score": score,
            "best_path": best_path,
        }

        meta_path = os.path.join(self.offline_dir, "offline_summary.json")
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=4)
    
    def save_startup_data(
        self, 
        episode_id: int,
        goal: str,
        max_steps: int,
        cfg
    ):
        data = {
            "episode_id": episode_id,
            "goal": goal,
            "max_steps": max_steps
        }
        meta_path = os.path.join(self.run_dir, "startup_summary.json")
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=4)
        with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

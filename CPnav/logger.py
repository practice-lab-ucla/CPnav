import os
import cv2
import json
from datetime import datetime
from typing import List, Tuple, Optional


def quaternion_to_wxyz(rotation):
    return [rotation.w, rotation.x, rotation.y, rotation.z]


class Logger:
    def __init__(self, root_dir: str = "./logs"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(root_dir, f"run_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.step = 0

        # subfolders
        self.img_dir = os.path.join(self.run_dir, "images")
        self.txt_dir = os.path.join(self.run_dir, "texts")
        self.json_dir = os.path.join(self.run_dir, "meta")

        for d in [self.img_dir, self.txt_dir, self.json_dir]:
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
                {"index": i, "r": float(r), "theta": float(theta)}
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
        total_distance: float,
        start_xz: Tuple[float, float],
        end_xz: Tuple[float, float],
        geodesic_distance: float,
        last_image
    ):
        """
        Saves final episode metadata and the terminal image.
        """
        # 1. Save Terminal Image
        if last_image is not None:
            term_img_path = os.path.join(self.img_dir, "terminal.png")
            # Convert RGB to BGR for OpenCV
            cv2.imwrite(term_img_path, last_image[:, :, ::-1])

        # 2. Save Metadata JSON
        data = {
            "goal": goal_text,
            "steps": step_count,
            "total_distance_traversed": total_distance,
            "start_position_xz": start_xz,
            "end_position_xz": end_xz,
            "displacement_error": geodesic_distance, # Start-to-End displacement
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        meta_path = os.path.join(self.json_dir, "terminal_summary.json")
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=4)
            
        print(f"[Logger] Terminal data saved to {self.run_dir}")
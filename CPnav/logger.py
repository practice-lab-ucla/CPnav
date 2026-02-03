import os
import cv2
import json
from datetime import datetime
from typing import List, Tuple, Optional


def quaternion_to_wxyz(rotation):
    return [rotation.w, rotation.x, rotation.y, rotation.z]


class GPTStepLogger:
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
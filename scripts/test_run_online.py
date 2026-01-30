import cv2
from CPnav.env import ObjectNavEnv
# import matplotlib.pyplot as plt
from CPnav.action_proposer.action_proposer import ActionProposer
from CPnav.projection_on_image import project_actions_on_image  
from CPnav.action_proposer.voxel_map import VoxelMap

from CPnav.agent.gpt_agent import GPTAgent, query_stop_and_action
from CPnav.sim_wrapper import PolarAction, SimWrapper, NULL_ACTION


import yaml

import os
import asyncio
from dotenv import load_dotenv

import os
import json
import cv2
from datetime import datetime
from typing import List, Tuple, Optional


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
        # OpenCV expects BGR
        cv2.imwrite(img_path, overlay_rgb[:, :, ::-1])

        # ---------- texts ----------
        with open(os.path.join(self.txt_dir, f"{step_id}_stop.txt"), "w") as f:
            f.write(stop_text)

        with open(os.path.join(self.txt_dir, f"{step_id}_action.txt"), "w") as f:
            f.write(act_text)

        # ---------- structured metadata ----------
        meta = {
            "step": self.step,
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


async def main():
    load_dotenv(dotenv_path=".env")

    with open("config/CPNav.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    env = ObjectNavEnv(cfg)

    env._initialize_experiment()
    env._initialize_episode(0)

    obs = env.reset()

    rgba = obs["color_sensor"]          # shape (H, W, 4), uint8
    rgb = rgba[:, :, :3].copy()         # drop alpha channel and copy

    ap = ActionProposer(cfg)
    voxel = VoxelMap(cfg)

    api_key = os.getenv("OPENAI_API_KEY")
    gpt_agent = GPTAgent(model="gpt-4o-mini", api_key=api_key)

    logger = GPTStepLogger("./logs")

    while True:
        print("step", env.step_count)
        a_initial_raw = ap._build_initial_actions(obs)          
        a_after_explored = ap._refine_explored(a_initial_raw, obs["agent_state"], voxel)

        a_final_full  = ap._refine_actions(a_after_explored)       
        actions = [(r, theta) for (r, theta, _) in a_final_full]

        agent_state = obs["agent_state"]
        sensor_state = agent_state.sensor_states["color_sensor"]

        rgb = obs["color_sensor"][:, :, :3]          # shape (H, W, 4), uint8
        overlay_rgb = rgb.copy()         # drop alpha channel and copy

        projected_mapping = project_actions_on_image(
            cfg,
            overlay_rgb,
            actions,                # refined actions (same as before)
            agent_state,
            sensor_state,
            chosen_index=None,
            initial_actions=a_initial_raw,  
            show_initial=True
        )

        # === update voxel map and visualize it ===
        # Use a_initial_raw for voxel map update (bounded by max_action_dist)
        voxel.update_from_actions(
            a_initial_raw,
            agent_state,
            max_action_dist=ap.max_action_dist,
            clip_frac=ap.clip_frac,
        )

        # Use the *projected* actions to draw arrows/numbers on top-down map
        voxel_image = voxel.generate_voxel_image(
            projected_mapping,
            agent_state,
            chosen_action=None,   # or e.g. 1 if you want to highlight action "1"
        )


        stop_res, action_res = await query_stop_and_action(gpt_agent, overlay_rgb, goal="toilet", num_actions=len(actions))

        stop_text, stop_data, stop_payload_log, stop_prompt = stop_res
        act_text,  act_data,  act_payload_log,  act_prompt  = action_res

        done = stop_data.get("done", 0)
        gsv  = stop_data.get("global_semantic_score", None)
        action_idx = act_data.get("action", None)
        scores = act_data.get("confidence_score", None)

        logger.log_step(
            overlay_rgb=overlay_rgb,
            stop_text=stop_text,
            act_text=act_text,
            actions=actions,
            action_idx=action_idx,
            confidence_scores=scores,
            done=done,
            global_semantic_score=gsv,
        )

        if done:
            return

        action_idx = min(max(action_idx - 1, 0), len(actions))
        action = PolarAction(*actions[action_idx], "")

        obs = env.step(action)


if __name__ == "__main__":
    asyncio.run(main())
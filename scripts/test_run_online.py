import os
import yaml
import asyncio
import numpy as np
from dotenv import load_dotenv

from CPnav.env.env import ObjectNavEnv
from CPnav.action_proposer.voxel_map import VoxelMap
from CPnav.action_proposer.projection_on_image import project_actions_on_image
from CPnav.action_proposer.action_proposer import ActionProposer
from CPnav.agent.gpt_agent import GPTAgent, query_stop_and_action
from CPnav.env.sim_wrapper import PolarAction, SimWrapper, NULL_ACTION
from CPnav.logger import GPTStepLogger


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

    for _ in range(50):
        print("step", env.step_count)
        print("agent position", obs["agent_state"].position)
        print("agent rotation", obs["agent_state"].rotation)
        agent_state = obs["agent_state"]
        a_initial_raw = ap._build_initial_actions(obs)          
        a_after_explored = ap._refine_explored(a_initial_raw, agent_state, voxel)

        a_final_full  = ap._refine_actions(a_after_explored)       
        actions = [(r, theta) for (r, theta, _) in a_final_full]

        sensor_state = agent_state.sensor_states["color_sensor"]

        rgb = obs["color_sensor"][:, :, :3]          # shape (H, W, 4), uint8
        overlay_rgb = rgb.copy()         # drop alpha channel and copy

        actions_on_image = project_actions_on_image(
            cfg,
            overlay_rgb,
            actions,                # refined actions (same as before)
            agent_state,
            sensor_state,
            chosen_index=None,
            initial_actions=a_initial_raw,  
            show_initial=False
        )

        sorted_actions = [
            PolarAction(r, theta)
            for (r, theta), _
            in sorted(actions_on_image.items(), key=lambda kv: kv[1])
        ]

        # === update voxel map and visualize it ===
        # Use a_initial_raw for voxel map update (bounded by max_action_dist)

        # Use the *projected* actions to draw arrows/numbers on top-down map
        voxel_image = voxel.generate_voxel_image(
            actions_on_image,
            agent_state,
            chosen_action=None,   # or e.g. 1 if you want to highlight action "1"
        )

        voxel.update_from_actions(
            a_initial_raw,
            agent_state,
            max_action_dist=ap.max_action_dist,
            clip_frac=ap.clip_frac,
        )

        stop_res, action_res = await query_stop_and_action(gpt_agent, overlay_rgb, goal="toilet", num_actions=len(actions))

        stop_text, stop_data, stop_payload_log, stop_prompt = stop_res
        act_text,  act_data,  act_payload_log,  act_prompt  = action_res

        done = stop_data.get("done", 0)
        gsv  = stop_data.get("global_semantic_score", None)
        action_selected = act_data.get("action", None)
        scores = act_data.get("confidence_score", None)

        logger.log_step(
            overlay_rgb=overlay_rgb,
            voxel_rgb=voxel_image,
            stop_text=stop_text,
            act_text=act_text,
            actions=actions,
            action_idx=action_selected,
            confidence_scores=scores,
            done=done,
            global_semantic_score=gsv,
        )

        if done:
            return
        
        if action_selected == 0:
            action = PolarAction(0, np.pi/2, "turn around")
            obs = env.step(action)
        
        else:
            action_idx = min(max(action_selected-1, 0), len(actions))
            action = sorted_actions[action_idx]

            obs = env.step(action)
        
        print("GPT choose action number", action_selected)
        print(f"executed action r={action.r}, theta={action.theta}")


if __name__ == "__main__":
    asyncio.run(main())
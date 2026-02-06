import numpy as np
from habitat_sim.utils.common import quat_to_angle_axis

import os
import yaml
import math
import asyncio
import numpy as np
from dotenv import load_dotenv
from typing import Optional, Tuple

from CPnav.env import ObjectNavEnv
from CPnav.action_proposer.voxel_map import VoxelMap
from CPnav.projection_on_image import project_actions_on_image
from CPnav.action_proposer.action_proposer import ActionProposer
from CPnav.agent.gpt_agent import GPTAgent, query_stop_and_action
from CPnav.sim_wrapper import PolarAction
from CPnav.logger import GPTStepLogger
from CPnav.tree import ExplorationTree, Pose2D
from CPnav.utils import quat_to_yaw_habitat
from CPnav.offline_bfs import OfflineGridTree


def record_node(tree, step_id: int, obs, sorted_actions, gsv, scores):
    s = obs["agent_state"]
    pos = s.position
    yaw = quat_to_yaw_habitat(s.rotation)

    tree.add_step(
        step_id=step_id,
        pose=Pose2D(pos[0], pos[2], yaw),
        actions=tree.to_action_proposal(sorted_actions),
        gsv=gsv,
        scores=scores,
    )


async def main():
    load_dotenv(dotenv_path=".env")

    with open("config/CPNav.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    env = ObjectNavEnv(cfg)
    tree = ExplorationTree()
    offline_tree = OfflineGridTree(float(cfg["agent_cfg"]["grid_spacing_m"]))

    env._initialize_experiment()
    goal = env._initialize_episode(0)

    obs = env.reset()

    ap = ActionProposer(cfg)
    voxel = VoxelMap(cfg)

    api_key = os.getenv("OPENAI_API_KEY")
    gpt_agent = GPTAgent(model="gpt-4o-mini", api_key=api_key)

    logger = GPTStepLogger("./logs")

    next_node_id = 0
    cur_node = next_node_id
    next_node_id += 1

    while env.step_count < 50:
        agent_state = obs["agent_state"]

        # --- build actions ---
        a_initial_raw = ap._build_initial_actions(obs)
        a_after_explored = ap._refine_explored(a_initial_raw, agent_state, voxel)
        a_final_full = ap._refine_actions(a_after_explored)
        actions = [(r, theta) for (r, theta, _) in a_final_full]

        sensor_state = agent_state.sensor_states["color_sensor"]
        overlay_rgb = obs["color_sensor"][:, :, :3].copy()

        actions_on_image = project_actions_on_image(
            cfg,
            overlay_rgb,
            actions,
            agent_state,
            sensor_state,
            chosen_index=None,
            initial_actions=a_initial_raw,
            show_initial=False,
        )

        sorted_actions = [
            PolarAction(r, theta)
            for (r, theta), _
            in sorted(actions_on_image.items(), key=lambda kv: kv[1])
        ]

        voxel_image = voxel.generate_voxel_image(
            actions_on_image, agent_state, chosen_action=None
        )

        voxel.update_from_actions(
            a_initial_raw,
            agent_state,
            max_action_dist=ap.max_action_dist,
            clip_frac=ap.clip_frac,
        )

        # --- query GPT ---
        stop_res, action_res = await query_stop_and_action(
            gpt_agent, overlay_rgb, goal=goal, num_actions=len(sorted_actions)
        )

        stop_text, stop_data, _, _ = stop_res
        act_text, act_data, _, _ = action_res

        done = int(stop_data.get("done", 0))
        gsv = stop_data.get("global_semantic_score", None)
        action_selected = act_data.get("action", None)  # 0..N (or None)
        scores = act_data.get("confidence_score", None)

        logger.log_step(
            overlay_rgb=overlay_rgb,
            voxel_rgb=voxel_image,
            position=obs["agent_state"].position,
            rotation=obs["agent_state"].rotation,
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

        # 1) Record/update the node you are CURRENTLY at
        record_node(tree, cur_node, obs, sorted_actions, gsv, scores)

        tree.debug_print_node(0)

        # --------------------------
        # Decide + execute movement
        # --------------------------
        N = len(sorted_actions)


        if action_selected == 0 and cur_node == 0:
            print("step 0 special: rotate 180 and advance")

            # create new node id
            new_node = next_node_id
            next_node_id += 1

            # link 0 -> new_node
            tree.link_parent(cur_node, new_node)

            # execute 180-degree rotation (in-place)
            obs = env.step(PolarAction(0.0, np.pi))

            # advance current node
            cur_node = new_node


        elif action_selected == 0:
            # rewind
            local = tree.local_backtrack(cur_node)
            if local is not None:
                target_step, target_action_id = local
            else:
                result = tree.global_backtrack(next_node_id-1)
                if result is None:
                    print("No more action to backtrack!")
                    return
                target_step, target_action_id = result

            # (A) teleop to existing target node (does NOT create a new node)
            obs = env.step(tree.get_node_position(target_step))
            prev_node = cur_node
            cur_node = target_step

            # (B) mark tried on the node we act FROM
            tree.mark_tried(cur_node, target_action_id)

            # (C) execute that action -> creates a NEW node
            new_node = next_node_id
            next_node_id += 1

            tree.link_parent(cur_node, new_node)

            xz_prev = (obs["agent_state"].position[0], obs["agent_state"].position[2])
            action_score = tree.get_action_score(cur_node, target_action_id)
            action = tree.get_node_action(cur_node, target_action_id)
            
            obs = env.step(action)
            
            xz_new = (obs["agent_state"].position[0], obs["agent_state"].position[2])
            offline_tree.add_edge(xz_prev, xz_new, action_score)
            cur_node = new_node

            print("step", prev_node, "rewind -> step", target_step, "action", target_action_id)


        else:
            action_selected = int(max(1, min(action_selected, N)))
            print("step", cur_node, "execute action", action_selected)

            tree.mark_tried(cur_node, action_selected)

            action_score = tree.get_action_score(cur_node, action_selected)
            action = sorted_actions[action_selected - 1]

            new_node = next_node_id
            next_node_id += 1

            xz_prev = (obs["agent_state"].position[0], obs["agent_state"].position[2])
            tree.link_parent(cur_node, new_node)
            obs = env.step(action)
            xz_new = (obs["agent_state"].position[0], obs["agent_state"].position[2])
            offline_tree.add_edge(xz_prev, xz_new, action_score)
            cur_node = new_node


if __name__ == "__main__":
    asyncio.run(main())

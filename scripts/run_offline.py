import asyncio
import numpy as np
import os

from CPnav.runner import Runner
from CPnav.env import PolarAction


async def run_offline_search(runner: Runner):
    """
    Main Loop: Continually picks the best 'Global Frontier' and explores
    a new branch from there until the step budget is exhausted.
    """
    while runner.has_steps():
        # 1. Frontier Selection: Get best global action from Priority Queue
        #    Expected return: (step_id, action_id) based on your updated tree code
        #    Note: 'score' might need to be fetched separately if not returned by get_best_global_action
        frontier = runner.tree.get_best_global_action() 
        
        if not frontier:
            print("No more valid frontiers to explore.")
            break
            
        step_id, action_id = frontier
        
        # 2. Teleportation: Reset agent to the frontier node
        #    (Using a helper method on runner is cleaner than raw env.step)
        target_pose = runner.tree.get_node_position(step_id)
        obs = runner.teleport_agent(target_pose) 

        # 3. Branch Initialization: Execute the specific queued action 
        #    to "break out" of the existing tree into new space.
        action_primitive = runner.tree.get_node_action(step_id, action_id)
        score = runner.tree.get_action_score(step_id, action_id)
        
        # Mark as tried immediately so we don't pick it again
        runner.tree.mark_tried(step_id, action_id)

        obs = runner.execute_move(action_primitive, obs, score)

        # 4. Branch Exploration: Hand off to the local loop
        #    This runs until the agent backtracks, finishes, or runs out of steps.
        obs = await explore_branch(runner, obs)
        
        if runner.is_mission_complete():
            return runner.get_result()

    # Budget exhausted or graph fully explored
    return runner.offline_tree.get_best_path_by_score()


async def explore_branch(runner: Runner, obs):
    """
    Sub-Loop: Standard VLM navigation (Perceive -> Decide -> Act)
    continues until a stopping condition is met.
    """
    while runner.has_steps():
        # A. Check Early Exit: Have we stumbled into an area we already mapped?
        pos = obs["agent_state"].position
        if runner.offline_tree.grid_explored((pos[0], pos[2])):
            print("Branch merged into explored territory. Stopping branch.")
            break

        # B. Perceive
        overlay, voxel_img, sorted_actions, actions = runner.perceive(obs)

        # C. Decide
        decision = await runner.decide(overlay, sorted_actions)

        # D. Log
        runner.log_and_record(obs, decision, overlay, voxel_img, actions, sorted_actions)

        # E. Handle High-Level Outcomes
        if decision.done:
            runner.record_goal(obs)
            break
        
        if decision.action_idx == 0:
            # "Backtrack" / Turn-around means this branch is dead.
            # We return to the main loop to pick a NEW global frontier.
            break

        # F. Act (Move Forward)
        idx = int(max(1, min(decision.action_idx, len(sorted_actions))))
        runner.tree.mark_tried(runner.cur_node_id, idx)

        action = sorted_actions[idx - 1]
        score = runner.tree.get_action_score(runner.cur_node_id, idx)

        obs = runner.execute_move(action, obs, score)
    
    return obs


async def main():
    runner = Runner("config/CPnav_5_nano.yaml")
    obs = runner.initialize()

    while runner.has_steps():
        # 1. Perceive
        overlay, voxel_img, sorted_actions, actions = runner.perceive(obs)

        # 2. Decide
        decision = await runner.decide(overlay, sorted_actions)

        # 3. Log
        runner.log_and_record(obs, decision, overlay, voxel_img, actions, sorted_actions)

        # 4. Handle "Done" Logic (Your specific request)
        if decision.done:
            print("Done condition met. Running post-process...")
            runner.record_goal(obs)
            await run_offline_search(runner)

        # 5. Act
        if decision.action_idx == 0:
            if runner.cur_node_id == 0:
                # Special Start
                obs = runner.execute_move(PolarAction(0.0, np.pi), obs)
            else:
                # Backtrack
                obs = runner.execute_backtrack()
                if obs is None: break
        else:
            # Normal Move
            idx = int(max(1, min(decision.action_idx, len(sorted_actions))))
            runner.tree.mark_tried(runner.cur_node_id, idx)
            
            action = sorted_actions[idx - 1]
            score = runner.tree.get_action_score(runner.cur_node_id, idx)
            
            obs = runner.execute_move(action, obs, score)

if __name__ == "__main__":
    asyncio.run(main())
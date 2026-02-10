import asyncio
import numpy as np
import os

from CPnav.runner import Runner
from CPnav.env import PolarAction



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
            return

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
import asyncio
import argparse
from CPnav.runner import Runner


async def run_offline_search(runner: Runner):
    """
    Main Loop: Continually picks the best 'Global Frontier' and explores
    a new branch from there until the step budget is exhausted.
    """
    score_threshold = None
    while runner.has_steps():
        # 1. Frontier Selection: Get best global action from Priority Queue
        #    Expected return: (step_id, action_id) based on your updated tree code
        #    Note: 'score' might need to be fetched separately if not returned by get_best_global_action
        frontier = runner.tree.get_best_global_action() 
        
        if not frontier:
            print("No more valid frontiers to explore.")
            break
            
        step_id, action_id, score = frontier

        if isinstance(score_threshold, float):
            if score < score_threshold:
                print("Explored all node")
                break
        
        # 2. Teleportation: Reset agent to the frontier node
        #    (Using a helper method on runner is cleaner than raw env.step)
        target_pose = runner.tree.get_node_position(step_id)

        # avoid backtrack to goal node
        if runner.offline_tree.is_goal((target_pose.x, target_pose.z)):
            continue

        obs = runner.teleport_agent(target_pose) 

        # 3. Branch Initialization: Execute the specific queued action 
        #    to "break out" of the existing tree into new space.
        action_primitive = runner.tree.get_node_action(step_id, action_id)
        score = runner.tree.get_action_score(step_id, action_id)
        
        # Mark as tried immediately so we don't pick it again
        runner.tree.mark_tried(step_id, action_id)
        print(f"Expanding: Node {step_id} Action {action_id}")

        obs = runner.execute_move(action_primitive, obs, score)

        # 4. Branch Exploration: Hand off to the local loop
        #    This runs until the agent backtracks, finishes, or runs out of steps.
        obs, score_threshold = await explore_branch(runner, obs)

    # Budget exhausted or graph fully explored
    score, best_path = runner.offline_tree.get_best_path_by_score()
    runner.record_offline_complete(score, best_path)
    return score


async def explore_branch(runner: Runner, obs):
    """
    Sub-Loop: Standard VLM navigation (Perceive -> Decide -> Act)
    continues until a stopping condition is met.
    """
    while runner.has_steps():
        # A. Perceive
        raw, overlay, voxel_img, sorted_actions, actions = runner.perceive(obs)

        # B. Decide
        decision = await runner.decide(raw, overlay, sorted_actions)

        # C. Log
        runner.log_and_record(obs, decision, overlay, voxel_img, actions, sorted_actions)
        
        # D. Handle High-Level Outcomes
        if decision.done:
            runner.record_goal_offline(obs)
            break
        
        if (decision.action_idx == 0) or (len(sorted_actions) == 0) or (decision.action_idx is None):
            # "Backtrack" / Turn-around means this branch is dead.
            # We return to the main loop to pick a NEW global frontier.
            break

        # E. Act (Move Forward)
        idx = int(max(1, min(decision.action_idx, len(sorted_actions))))
        obs = runner.execute_forward(idx, obs)

        # F. Check Early Exit: Have we stumbled into an area we already mapped?
        pos = obs["agent_state"].position
        if runner.offline_tree.grid_explored((pos[0], pos[2])):
            break
    
    return obs, runner.offline_tree.get_best_path_by_score()[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Run CPNav online + offline search")

    parser.add_argument(
        "--config",
        type=str,
        default="config/CPnav_4o_mini.yaml",
        help="Path to config YAML file",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/",
        help="Directory to save logs",
    )

    parser.add_argument(
        "--episode",
        type=int,
        default=100,
        help="Episode ID to initialize",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    runner = Runner(args.config, log_dir=args.log_dir)
    obs = runner.initialize(args.episode)

    print("===== RUN CONFIG =====")
    print("Config:", args.config)
    print("Log Path:", args.log_dir)
    print("Episode:", args.episode)
    print("======================")

    while runner.has_steps():
        # 1. Perceive
        raw, overlay, voxel_img, sorted_actions, actions = runner.perceive(obs)

        # 2. Decide
        decision = await runner.decide(raw, overlay, sorted_actions)
        
        # 3. Log
        runner.log_and_record(obs, decision, overlay, voxel_img, actions, sorted_actions)

        # 4. Handle "Done" Logic, do not record goal node
        if decision.done:
            print("Found goal online")
            runner.record_goal(obs)
            print("score is", await run_offline_search(runner))
            return

        # 5. Act
        if decision.action_idx is None:
            print("Warnning GPT does not give correct action")
        if (decision.action_idx == 0) or (len(sorted_actions) == 0) or (decision.action_idx is None):
            # Special Start
            obs = runner.execute_turnaround(obs)
        elif (decision.action_idx == 0) or (len(sorted_actions) == 0):
            # Backtrack
            obs = runner.execute_backtrack()
            if obs is None: break
        else:
            # Normal Move
            idx = int(max(1, min(decision.action_idx, len(sorted_actions))))
            obs = runner.execute_forward(idx, obs)

if __name__ == "__main__":
    asyncio.run(main())
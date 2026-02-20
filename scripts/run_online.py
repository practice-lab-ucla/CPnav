import asyncio
import argparse
from CPnav.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="Run CPNav online")

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for backtrack action"
    )

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

    parser.add_argument(
        "--mode",
        type=str,
        default="cp_nav",  # Updated from 100 to a valid default
        choices=["cp_nav", "simple_set", "prompt_set", "ensemble_set"],
        help="Navigation mode for the agent (default: cp_nav)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,  # Updated from 100 to a valid default
        help="Threshold used for constructing prediction set, only used if mode = simple_set or ensemble_set",
    )

    return parser.parse_args()


async def main():
    args = parse_args()
    threshold = float(args.threshold)
    if args.mode != "cp_nav":
        threshold = 0.0
    runner = Runner(
        args.config, 
        log_dir=args.log_dir, 
        threshold=threshold, 
        mode=args.mode, 
        alpha=float(args.alpha)
    )
    obs = runner.initialize(args.episode)

    print("===== RUN CONFIG =====")
    print("Config:", args.config)
    print("Log Path:", args.log_dir)
    print("Episode:", args.episode)
    print("Threshold:", args.threshold)
    print("Mode:", args.mode)
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
            return

        # 5. Act
        if decision.action_idx is None:
            print("Warnning GPT does not give correct action")
        if (decision.action_idx == 0) and (runner.cur_node_id == 0):
            # Special Start
            obs = runner.execute_turnaround(obs)
        elif (decision.action_idx == 0) or (len(sorted_actions) == 0) or (decision.action_idx is None):
            obs = runner.execute_backtrack()
            if obs is None: break
        else:
            # Normal Move
            idx = int(max(1, min(decision.action_idx, len(sorted_actions))))
            obs = runner.execute_forward(idx, obs)
    
    runner.record_failure(obs)

if __name__ == "__main__":
    asyncio.run(main())
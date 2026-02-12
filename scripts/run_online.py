import asyncio
import argparse
from CPnav.runner import Runner


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
            return

        # 5. Act
        if (decision.action_idx == 0) and (runner.cur_node_id == 0):
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
import os
import cv2
import yaml
import asyncio
import numpy as np
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

from CPnav.logger import Logger
from CPnav.agent import GPTAgent
from CPnav.utils import quat_to_yaw_habitat
from CPnav.env import ObjectNavEnv, PolarAction, TeleopAction
from CPnav.tree import ExplorationTree, OfflineGridTree, Pose2D, visualize_tree
from CPnav.action_proposer import VoxelMap, project_actions_on_image, ActionProposer

# --- Data Structure for GPT Decisions ---
@dataclass
class AgentDecision:
    stop_text: str
    act_text: str
    action_idx: Optional[int]
    done: bool
    gsv: Optional[float]
    confidence_scores: Optional[List[float]]

class Runner:
    def __init__(self, cfg_path: str = "config/CPnav_5_nano.yaml", log_dir="logs/"):
        # 1. Load Config & Env
        load_dotenv(dotenv_path=".env")
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)
            
        self.env = ObjectNavEnv(self.cfg)
        
        # 2. Init Agent
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
        model = self.cfg["agent_cfg"]["model"]
        self.gpt_agent = GPTAgent(
            model=model, api_key=api_key
        )

        # 3. Init State & Memory
        self.tree = ExplorationTree()
        self.offline_tree = OfflineGridTree(
            float(self.cfg["agent_cfg"]["grid_spacing_m"])
        )
        self.log_dir = log_dir
        
        self.ap = ActionProposer(self.cfg)
        self.voxel = VoxelMap(self.cfg)
        
        # Tracking Variables
        self.cur_node_id = 0
        self.next_node_id = 1
        self.goal = None # Will be string (goal text) later
        self.tried_step0_turnaround = False
        self.history = []
        self.start_xz = None
        self.goal_xz = None
        self.goal_image = None

        # NEW: Track total distance
        self.traversed_distance = 0.0

    def initialize(self, episode_id: int = 0):
        """Starts the environment and resets internal state."""
        self.goal = self.env.initialize_episode(episode_id)
        print("gpt using model", self.gpt_agent.model)
        print("initializing episode", episode_id, "to find", self.goal)
        
        obs = self.env.reset()
        start_position = obs["agent_state"].position
        self.start_xz = (start_position[0], start_position[2])
        self.offline_tree.initialize(obs)
        self.logger = Logger(episode_id, self.log_dir)
        self.logger.save_startup_data(
            episode_id=episode_id,
            goal=self.goal,
            max_steps=self.cfg["env_cfg"]["max_steps"],
            cfg=self.cfg
        )

        return obs

    def perceive(self, obs: Dict[str, Any]):
        """
        1. Proposes actions based on VoxelMap.
        2. Projects actions onto the current image.
        3. Generates the Voxel RGB visualization.
        """
        agent_state = obs["agent_state"]
        target_size = tuple(self.cfg["agent_cfg"].get("query_image_resolution", (1920,1080)))
        
        # Update Voxel & Propose
        actions = self.ap.propose_actions_and_update_voxel(obs, self.voxel)
        
        # Project on Image
        sensor_state = agent_state.sensor_states["color_sensor"]
        raw_rgb = obs["color_sensor"][:, :, :3]
        overlay_rgb = obs["color_sensor"][:, :, :3].copy()
        
        actions_on_image = project_actions_on_image(
            self.cfg, overlay_rgb, actions, agent_state, sensor_state, chosen_index=None
        )
        
        # Generate Voxel View
        voxel_image = self.voxel.generate_voxel_image(
            actions_on_image, agent_state, chosen_action=None
        )

        # Sort Actions (Polar format for the Tree)
        # sorting by the 'theta' (key[1]) or however your dict is structured
        sorted_actions = [
            PolarAction(r, theta)
            for (r, theta), _ in sorted(actions_on_image.items(), key=lambda kv: kv[1])
        ]

        raw_rgb_resized = cv2.resize(raw_rgb, target_size, interpolation=cv2.INTER_AREA)
        overlay_rgb_resize = cv2.resize(overlay_rgb, target_size, interpolation=cv2.INTER_AREA)
        
        return raw_rgb_resized, overlay_rgb_resize, voxel_image, sorted_actions, actions

    async def decide(self, raw_rgb, overlay_rgb, sorted_actions) -> AgentDecision:
        """
        Queries GPT to decide on stop condition and next action.
        """
        # Note: Ensure query_stop_and_action is available on self.gpt_agent 
        # or import the standalone function if it's not a method of GPTAgent.
        stop_res, action_res = await self.gpt_agent.query_stop_and_action(
            raw_rgb, overlay_rgb, goal=self.goal, num_actions=len(sorted_actions)
        )
        
        stop_text, stop_data, _, _ = stop_res
        act_text, act_data, _, _ = action_res
        
        return AgentDecision(
            stop_text=stop_text,
            act_text=act_text,
            action_idx=act_data.get("action", None),
            done=int(stop_data.get("done", 0)),
            gsv=stop_data.get("global_semantic_score", None),
            confidence_scores=act_data.get("confidence_score", None)
        )

    def log_and_record(self, 
        obs, 
        decision: AgentDecision, 
        overlay_rgb, 
        voxel_rgb, 
        actions, 
        sorted_actions
    ):
        """
        Logs the step to disk and records the node in the ExplorationTree.
        """
        # 1. Disk Log
        self.logger.log_step(
            overlay_rgb=overlay_rgb,
            voxel_rgb=voxel_rgb,
            position=obs["agent_state"].position,
            rotation=obs["agent_state"].rotation,
            stop_text=decision.stop_text,
            act_text=decision.act_text,
            actions=actions,
            action_idx=decision.action_idx,
            confidence_scores=decision.confidence_scores,
            done=decision.done,
            global_semantic_score=decision.gsv,
        )

        # 2. Tree Record
        self._record_to_tree(obs, sorted_actions, decision.gsv, decision.confidence_scores)
        
        # self.tree.debug_print_node(0)
        

    def execute_move(self, action, current_obs, action_score=None):
        """
        Executes a move, updates distance, offline grid, and links tree nodes.
        """
        # 1. Capture previous position
        prev_xz = self.get_xz(current_obs)
        
        # 2. Execute in Sim
        obs = self.env.step(action)
        
        # 3. Calculate Forward Distance
        curr_xz = self.get_xz(obs)
        dist = np.linalg.norm(np.array(curr_xz) - np.array(prev_xz))
        self.traversed_distance += dist
        
        # 4. Link Tree Nodes & Update Offline Grid
        new_node = self.next_node_id
        self.next_node_id += 1
        self.tree.link_parent(self.cur_node_id, new_node)
        
        if action_score is not None:
            self.offline_tree.add_edge(prev_xz, curr_xz, action_score)

        self.cur_node_id = new_node        
        return obs

    def execute_backtrack(self):
        """
        Backtracks to a parent node. Calculates the specific path distance required
        to traverse the tree back to that node and adds it to total distance.
        """
        # 1. Find Target
        local = self.tree.local_backtrack(self.cur_node_id)
        self.tree.mark_tried(self.cur_node_id, 0)
        if local:
            target_step, target_action_id = local
        else:
            res = self.tree.global_backtrack(self.next_node_id - 1)
            if not res:
                if self.tried_step0_turnaround:
                    print("No more actions to backtrack! Exploration exhausted.")
                    return None
                else:
                    target_pos = self.tree.get_node_position(0)  # execute the special turnaround action at 0
                    
                    # Calculate distance from current to root (0)
                    backtrack_dist = self.tree.get_path_length(self.cur_node_id, 0)
                    self.traversed_distance += backtrack_dist
                    obs = self.env.step(target_pos)
                    return self.execute_turnaround(obs)

            target_step, target_action_id = res

        print(f"Rewinding: Node {self.cur_node_id} -> Node {target_step} (Action {target_action_id})")
        self.history.append(f"{self.cur_node_id}->{target_step} {target_action_id}")

        # 2. Calculate Traversal Distance for this Backtrack
        # We use the tree to find the path length from current -> target
        backtrack_dist = self.tree.get_path_length(self.cur_node_id, target_step)
        self.traversed_distance += backtrack_dist

        # 3. Teleport
        target_pos = self.tree.get_node_position(target_step)
        obs = self.env.step(target_pos)
        
        # Update current pointer
        self.cur_node_id = target_step

        # 4. Mark Action & Execute next move
        self.tree.mark_tried(self.cur_node_id, target_action_id)
        
        action = self.tree.get_node_action(self.cur_node_id, target_action_id)
        score = self.tree.get_action_score(self.cur_node_id, target_action_id)
        
        return self.execute_move(action, obs, score)
    
    def execute_forward(self, action_idx, obs):
        """execute action_idx at current node"""
        self.tree.mark_tried(self.cur_node_id, action_idx)
        self.history.append(f"{self.cur_node_id}->{self.cur_node_id} {action_idx}")
        print(f"Forwarding: Node {self.cur_node_id} (Action {action_idx})")
        action = self.tree.get_node_action(self.cur_node_id, action_idx)
        score = self.tree.get_action_score(self.cur_node_id, action_idx)
        return self.execute_move(action, obs, score)
    
    def execute_turnaround(self, obs):
        self.tried_step0_turnaround = True
        self.history.append(f"{self.cur_node_id}->0 turnaround")
        print(f"Rewinding: Node {self.cur_node_id} -> Node 0 (Action turnaround)")
        return self.execute_move(PolarAction(0, np.pi), obs)

    def _record_to_tree(self, obs, sorted_actions, gsv, scores):
        """Internal helper to add a node to the ExplorationTree."""
        s = obs["agent_state"]
        pos = s.position
        yaw = quat_to_yaw_habitat(s.rotation)

        self.tree.add_step(
            step_id=self.cur_node_id,
            pose=Pose2D(pos[0], pos[2], yaw),
            actions=self.tree.to_action_proposal(sorted_actions),
            gsv=gsv,
            scores=scores,
        )
    
    def teleport_agent(self, pose: TeleopAction):
        """Abstraction for teleporting the agent."""
        # Assuming env.step accepts a TeleopAction or similar
        return self.env.step(pose)

    @staticmethod
    def get_xz(state_or_obs):
        """Extracts (x, z) tuple from an observation or agent state."""
        if isinstance(state_or_obs, dict):
            state = state_or_obs["agent_state"]
        else:
            state = state_or_obs
        return (state.position[0], state.position[2])
    
    def has_steps(self):
        return self.env.step_count < int(self.cfg["env_cfg"]["max_steps"])
    
    def record_goal(self, obs):
        """
        Finalizes the episode: records goal state and triggers terminal logging.
        """
        state = obs["agent_state"]
        xz = (state.position[0], state.position[2])
        self.offline_tree.add_goal(xz)
        
        self.goal_xz = xz
        self.goal_image = obs["color_sensor"][:, :, :3].copy()
        
        # Calculate 'Geodesic'
        if self.start_xz and self.goal_xz:
            displacement = self.env.start_to_goal_geodesic_distance(self.start_xz, self.goal_xz)
        else:
            displacement = 0.0

        # Trigger Logger
        if self.logger:
            self.logger.save_terminal_data(
                goal_text=str(self.goal),
                step_count=self.env.step_count if hasattr(self.env, 'step_count') else len(self.history),
                traversed_distance=self.traversed_distance,
                start_xz=self.start_xz,
                end_xz=self.goal_xz,
                geodesic_distance=displacement,
                history=self.history,
                last_image=self.goal_image
            )

            visualize_tree(
                self.offline_tree, 
                os.path.join(self.logger.offline_dir, "offline_tree.png")
            )
        
        print(f"Episode Completed. Total Distance: {self.traversed_distance:.2f}m")


    def record_goal_offline(self, obs):
        """
        Finalizes the episode: records terminal state for offline.
        """
        state = obs["agent_state"]
        xz = (state.position[0], state.position[2])
        self.offline_tree.add_goal(xz)
    
    def record_offline_complete(self, score, best_path):
        self.logger.save_offline_data(score, best_path)
        visualize_tree(
            self.offline_tree, 
            os.path.join(self.logger.offline_dir, "offline_tree_refined.png")
        )
    
    def record_failure(self, obs):
        """
        Finalizes the episode: records goal state and triggers terminal logging.
        """

        # Trigger Logger
        if self.logger:
            self.logger.save_failed_terminal_data(
                goal_text=str(self.goal),
                step_count=self.env.step_count if hasattr(self.env, 'step_count') else len(self.history),
                traversed_distance=self.traversed_distance,
                start_xz=self.start_xz,
                history=self.history,
            )
        
        print(f"Episode Failed. Total Distance: {self.traversed_distance:.2f}m")
        

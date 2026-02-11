import os
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
    def __init__(self, cfg_path: str = "config/CPnav_5_nano.yaml"):
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
        self.logger = None
        self.ap = ActionProposer(self.cfg)
        self.voxel = VoxelMap(self.cfg)
        
        # 4. Tracking Variables
        self.cur_node_id = 0
        self.next_node_id = 1
        self.goal = Logger("./logs")
        self.tried_step0_turnaround = False
        self.history = []
        self.start_xz = None

    def initialize(self, episode_id: int = 0):
        """Starts the environment and resets internal state."""
        self.goal = self.env.initialize_episode(episode_id)
        print("gpt using model", self.gpt_agent.model)
        print("initializing episode", episode_id, "to find", self.goal)
        
        obs = self.env.reset()
        start_position = obs["agent_state"].position
        self.start_xz = (start_position[0], start_position[2])
        self.offline_tree.initialize(obs)
        return obs

    def perceive(self, obs: Dict[str, Any]):
        """
        1. Proposes actions based on VoxelMap.
        2. Projects actions onto the current image.
        3. Generates the Voxel RGB visualization.
        """
        agent_state = obs["agent_state"]
        
        # Update Voxel & Propose
        actions = self.ap.propose_actions_and_update_voxel(obs, self.voxel)
        
        # Project on Image
        sensor_state = agent_state.sensor_states["color_sensor"]
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
        
        return overlay_rgb, voxel_image, sorted_actions, actions

    async def decide(self, overlay_rgb, sorted_actions) -> AgentDecision:
        """
        Queries GPT to decide on stop condition and next action.
        """
        # Note: Ensure query_stop_and_action is available on self.gpt_agent 
        # or import the standalone function if it's not a method of GPTAgent.
        stop_res, action_res = await self.gpt_agent.query_stop_and_action(
            overlay_rgb, goal=self.goal, num_actions=len(sorted_actions)
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
                       sorted_actions):
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

        # 3. Save offline tree info for debugging
        visualize_tree(
            self.offline_tree, 
            os.path.join(self.logger.img_dir, f"{self.cur_node_id:04d}_offline_tree.png")
        )
        
        # self.tree.debug_print_node(0)

    def execute_move(self, action, current_obs, action_score=None):
        """
        Executes a move, updates the offline grid, and links the tree nodes.
        Returns: new_obs
        """
        # Capture previous position from the PASSED observation (safer)
        prev_xz = self.get_xz(current_obs)
        
        # Execute in Sim
        obs = self.env.step(action)
        
        # Link Tree Nodes
        new_node = self.next_node_id
        self.next_node_id += 1
        self.tree.link_parent(self.cur_node_id, new_node)
        
        # Update Offline Grid Map
        curr_xz = self.get_xz(obs)
        if action_score is not None:
            self.offline_tree.add_edge(prev_xz, curr_xz, action_score)

        # Update Internal Pointer
        self.cur_node_id = new_node        
        return obs

    def execute_backtrack(self):
        """
        Performs the backtracking logic:
        1. Identifies the target parent node (local or global).
        2. Teleports agent there.
        3. Executes the next untried action from that node.
        """
        # 1. Find Target
        local = self.tree.local_backtrack(self.cur_node_id)
        if local:
            target_step, target_action_id = local
        else:
            # Fallback to global backtrack
            # Passing next_node_id - 1 essentially implies checking from the most recent node backwards
            res = self.tree.global_backtrack(self.next_node_id - 1)
            if not res:
                if self.tried_step0_turnaround:
                    print("No more actions to backtrack! Exploration exhausted.")
                    return None
                else:
                    # try turn around action at step 0
                    target_pos = self.tree.get_node_position(0)
                    self.cur_node_id = 0
                    obs = self.env.step(target_pos)

                    print(f"Rewinding: Node {self.cur_node_id} -> Node {target_step} (Action turnaround)")
                    self.history.append(f"{self.cur_node_id}->{target_step} turnaround")

                    return self.execute_move(PolarAction(0.0, np.pi), obs)

            target_step, target_action_id = res

        print(f"Rewinding: Node {self.cur_node_id} -> Node {target_step} (Action {target_action_id})")
        self.history.append(f"{self.cur_node_id}->{target_step} {target_action_id}")

        # 2. Teleport
        # We assume env.step() can take a position vector for teleportation
        target_pos = self.tree.get_node_position(target_step)
        obs = self.env.step(target_pos)
        
        # Update current pointer
        self.cur_node_id = target_step

        # 3. Mark Action & Execute
        self.tree.mark_tried(self.cur_node_id, target_action_id)
        
        action = self.tree.get_node_action(self.cur_node_id, target_action_id)
        score = self.tree.get_action_score(self.cur_node_id, target_action_id)
        
        # Execute the move (this will create a NEW node branching off the old one)
        return self.execute_move(action, obs, score)
    
    def execute_forward(self, action_idx, obs):
        """execute action_idx at current node"""
        self.tree.mark_tried(self.cur_node_id, action_idx)
        self.history.append(f"{self.cur_node_id}->{self.cur_node_id} {action_idx}")
        action = self.tree.get_node_ation(self.cur_node_id, action_idx)
        score = self.tree.get_action_score(self.cur_node_id, action_idx)
        return self.execute_move(action, obs, score)
    
    def execute_turnaround(self, obs):
        self.tried_step0_turnaround = True
        self.history.append(f"{self.cur_node_id}->{self.cur_node_id} turnaround")
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
        state = obs["agent_state"]
        xz = (state.position[0], state.position[2])
        self.offline_tree.add_goal(xz)

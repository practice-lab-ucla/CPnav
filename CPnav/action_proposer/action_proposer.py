import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from .navigability_raycast import DepthNavigabilityRaycaster
from ..utils import local_to_global

# Type alias for clarity: (radius, theta, is_unexplored)
ActionCandidate = Tuple[float, float, bool]

class ActionProposer:
    """
    Generates candidate navigation actions (r, theta) based on depth observations
    and exploration heuristics.

    Pipeline:
    1. Raycast on Depth Image -> Initial (r, theta)
    2. Check Voxel Map -> Mark as Explored/Unexplored
    3. Filter & space out actions based on 'hard_spacing' and 'explore_bias'.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        
        # 1. Unpack Configs
        self.agent_cfg = cfg["agent_cfg"]
        self.ap_cfg = self.agent_cfg["action_proposer_cfg"]
        self.sim_cfg = cfg["sim_cfg"]

        # 2. Load Proposer Parameters
        self.num_theta = self.ap_cfg["num_theta"]
        self.max_action_dist = self.ap_cfg["max_action_dist"]
        self.min_action_dist = self.ap_cfg["min_action_dist"]
        self.clip_frac = self.ap_cfg["clip_frac"]
        self.hard_spacing = np.deg2rad(self.ap_cfg["hard_spacing_deg"])
        self.propose_if_all_explored = self.ap_cfg.get("propose_if_all_explored", False)
        self.enable_voxel = self.ap_cfg.get("enable_voxel", True)

        # Exploration Bias
        if self.ap_cfg.get("enable", False):
            self.explore_bias = self.ap_cfg.get("weight", 1.0)
        else:
            self.explore_bias = 0.0

        # 3. Precompute Angles based on Sensor FOV
        sensor_fov_deg = self.sim_cfg["sensor_cfg"]["fov"]
        # Use 1.5x FOV range for safety margin at edges, similar to original logic
        half_range_rad = np.deg2rad(sensor_fov_deg) * 1.5 / 2.0
        self.theta_values = np.linspace(-half_range_rad, half_range_rad, self.num_theta)

        # 4. Initialize Raycaster (Reused across steps)
        self.raycaster = DepthNavigabilityRaycaster(self.cfg)

    def propose_actions_and_update_voxel(self, obs: Dict[str, Any], voxel) -> List[Tuple[float, float]]:
        """
        Main entry point.
        Returns a list of (radius, theta) tuples for safe navigation.
        """
        # 1. Depth-based candidates
        a_initial = self._build_initial_actions(obs)
        
        # 2. Check against Voxel Map (Explored vs Unexplored)
        if self.enable_voxel:
            a_checked = self._check_explored_status(a_initial, obs["agent_state"], voxel)
        else:
            a_checked = [(r, theta, True) for r, theta in a_initial]
        
        # 3. Filter and refine based on spacing/bias
        a_final = self._refine_actions(a_checked)

        if not a_final and self.propose_if_all_explored:
            # If all actions are explored but config allows proposing anyway, 
            # then we can relax the filter to include explored actions.
            a_final = self._refine_actions([(r, theta, True) for (r, theta, _) in a_checked])

        # 4. Update the voxel map with the *raw* depth data
        #    (We use the initial raw actions to update the map, not the refined ones)
        if voxel is not None:
            voxel.update_from_actions(
                a_initial,
                obs["agent_state"],
                max_action_dist=self.max_action_dist,
                clip_frac=self.clip_frac,
            )

        # Return just (r, theta), dropping the 'explored' flag
        return [(r, theta) for (r, theta, _) in a_final]

    def _build_initial_actions(self, obs: Dict[str, Any]) -> List[Tuple[float, float]]:
        """
        Uses the Raycaster to convert Depth image -> Polar free-space rays.
        """
        return self.raycaster.build_initial_actions(obs, self.theta_values)

    def _check_explored_status(self, 
                               a_initial_raw: List[Tuple[float, float]], 
                               agent_state, 
                               voxel) -> List[ActionCandidate]:
        """
        Annotates each action with an 'is_unexplored' boolean by checking the VoxelMap.
        """
        if not a_initial_raw:
            return []

        # If voxel map isn't ready, assume everything is unexplored (safer to explore)
        if getattr(voxel, "init_pos", None) is None:
            return [(r, theta, True) for (r, theta) in a_initial_raw]

        # Prepare Top-Down Map
        # Merge the 'explored' visualization into the 'obstacle' map for checking
        topdown_map = voxel.voxel_map.copy()
        h, w, _ = topdown_map.shape
        
        try:
            # Overwrite areas that are explicitly marked as 'explored'
            mask = np.all(voxel.explored_map == voxel.explored_color, axis=-1)
            topdown_map[mask] = voxel.explored_color
        except ValueError:
            pass # Dimension mismatch, ignore explored overlay

        out = []
        for r, theta in a_initial_raw:
            # Check a point slightly closer than the max reach (clip_frac)
            check_dist = min(r * self.clip_frac, self.max_action_dist)
            
            # 1. Project to Local 3D (x, y, z) - Habitat frame: -Z is forward, Y is up
            local_pt = np.array([
                check_dist * np.sin(theta), 
                0.0, 
                -check_dist * np.cos(theta)
            ])

            # 2. Project to Global Grid
            try:
                global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
                gx, gy = voxel._global_to_grid(global_pt)
            except Exception:
                # If projection fails, treat as explored (don't prioritize it)
                out.append((r, theta, False))
                continue

            # 3. Check Neighborhood in Grid
            is_unexplored = self._is_neighborhood_unexplored(gx, gy, topdown_map, voxel.explored_color)
            out.append((r, theta, is_unexplored))

        return out

    def _is_neighborhood_unexplored(self, gx, gy, topdown_map, explored_color, radius=2) -> bool:
        """
        Returns True if the neighborhood has very few 'explored' pixels.
        """
        h, w, _ = topdown_map.shape
        gx, gy = int(gx), int(gy)
        
        # Clamp to bounds
        if not (0 <= gx < w and 0 <= gy < h):
            return False

        # Define slices
        x0, x1 = max(0, gx - radius), min(w - 1, gx + radius)
        y0, y1 = max(0, gy - radius), min(h - 1, gy + radius)

        # Count explored pixels in a cross pattern
        cnt = 0
        # Vertical strip
        col = topdown_map[y0:y1+1, gx]
        cnt += np.sum(np.all(col == explored_color, axis=-1))
        
        # Horizontal strip
        row = topdown_map[gy, x0:x1+1]
        cnt += np.sum(np.all(row == explored_color, axis=-1))

        # Threshold: if < 3 pixels are explored, we consider it unexplored
        return cnt < 3

    def _refine_actions(self, a_initial: List[ActionCandidate]) -> List[ActionCandidate]:
        """
        Selects the best actions based on distance, safety, and exploration bias.
        Filters out actions that lead to already explored areas.
        """
        if not a_initial:
            return []

        # 1. Filter: Keep only UNEXPLORED actions
        #    (We assume the agent wants to go where it hasn't been)
        candidates = []
        original_distances = {} # Map theta -> max radius found

        for r, theta, is_unexplored in a_initial:
            # Store the raw distance for lookup later
            # If multiple rays fall on same theta (rare with linspace), take min
            if theta not in original_distances:
                original_distances[theta] = r
            else:
                original_distances[theta] = min(original_distances[theta], r)

            if is_unexplored:
                # Apply clip_frac for the sorting/selection phase
                candidates.append((r * self.clip_frac, theta))

        if not candidates:
            return []

        # 2. Group by theta (redundancy check) and sort
        #    If multiple rays map to same theta, keep shortest (safest)
        unique_candidates = {}
        for r, theta in candidates:
            if theta not in unique_candidates or r < unique_candidates[theta][0]:
                unique_candidates[theta] = (r, theta, True)
        
        sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x[1])
        
        # Filter by min_dist safety
        valid_candidates = [x for x in sorted_candidates if x[0] > self.min_action_dist]
        
        if not valid_candidates:
            return []

        # 3. Apply Exploration Bias (Sweep Logic)
        final_selection = []
        selected_thetas = set()

        if self.explore_bias > 0:
            # A. Find the "Best" Ray (Longest Unexplored)
            best_ray = max(valid_candidates, key=lambda x: x[0])
            best_idx = valid_candidates.index(best_ray)
            
            final_selection.append(best_ray)
            selected_thetas.add(best_ray[1])

            # B. Sweep Forward (Angular spacing check)
            for i in range(best_idx + 1, len(valid_candidates)):
                curr = valid_candidates[i]
                if all(abs(curr[1] - t) > self.hard_spacing for t in selected_thetas):
                    final_selection.append(curr)
                    selected_thetas.add(curr[1])

            # C. Sweep Backward
            for i in range(best_idx - 1, -1, -1):
                curr = valid_candidates[i]
                if all(abs(curr[1] - t) > self.hard_spacing for t in selected_thetas):
                    final_selection.append(curr)
                    selected_thetas.add(curr[1])
            
            # D. Fill Gaps (using explore_bias weight)
            #    Allows adding rays that are closer than hard_spacing if weight allows
            spacing_threshold = self.hard_spacing * self.explore_bias
            for item in valid_candidates:
                if item[1] not in selected_thetas:
                    # Check distance to nearest selected
                    dist_to_nearest = min(abs(item[1] - t) for t in selected_thetas)
                    if dist_to_nearest > spacing_threshold:
                        final_selection.append(item)
                        selected_thetas.add(item[1])
        else:
            # No bias, take all valid candidates
            final_selection = valid_candidates

        # 4. Restore Original Distances
        #    We operated on clipped distances for safety logic, now give back the real 
        #    traversable distance (up to max_dist)
        out = []
        final_selection.sort(key=lambda x: x[1])
        
        for _, theta, is_unexplored in final_selection:
            raw_r = original_distances.get(theta, 0)
            final_r = min(raw_r, self.max_action_dist)
            out.append((final_r, theta, is_unexplored))

        return out
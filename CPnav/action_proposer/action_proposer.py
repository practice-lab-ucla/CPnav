# CPnav/action_proposer/action_proposer.py

import numpy as np
from .navigability_raycast import DepthNavigabilityRaycaster
from ..utils import local_to_global
import math


class ActionProposer:
    """
    CPNav-friendly reimplementation of the VLMNav `_action_proposer` logic.

    - Uses CPNav.yaml -> agent_cfg.action_proposer_cfg for hyperparams.
    - Uses DepthNavigabilityRaycaster to get lidar-like (r, theta) from RGBD.
    - For now, voxel_map is ignored (no map update yet).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        # Take the dedicated action proposer config block
        agent_cfg = cfg["agent_cfg"]
        sim_cfg = cfg["sim_cfg"]
        ap_cfg = agent_cfg["action_proposer_cfg"]

        # Basic radial discretization
        self.num_theta = ap_cfg["num_theta"]
        self.max_action_dist = ap_cfg["max_action_dist"]
        self.min_action_dist = ap_cfg["min_action_dist"]

        # Refinement knobs
        self.hard_spacing = np.deg2rad(ap_cfg["hard_spacing_deg"])  # radians
        self.clip_frac = ap_cfg["clip_frac"]


        # Explore bias
        if ap_cfg.get("enable", False):
            self.explore_bias = ap_cfg.get("weight", 1.0)
        else:
            self.explore_bias = 0.0

        # Angular range / sensor range (use camera FOV)
        sensor_fov_deg = sim_cfg["sensor_cfg"]["fov"]
        half_range_rad = np.deg2rad(sensor_fov_deg)* 1.5 / 2.0

        # Precompute theta samples in the camera FOV
        self.theta_values = np.linspace(-half_range_rad, half_range_rad, self.num_theta)

        # Depth-based raycaster helper: does RGBD → (r, theta)
        # Assumes DepthNavigabilityRaycaster(cfg) is how we constructed it.


    def _build_initial_actions(self, obs: dict):
        """
        Use DepthNavigabilityRaycaster to turn the current depth image into
        lidar-like radial distances:

            a_initial = [(r_i, theta_i), ...]

        where r_i is how far you can move along theta_i before hitting an obstacle.
        """
        # a_initial = [(self.max_action_dist, theta) for theta in self.theta_values]
        
        self.raycaster = DepthNavigabilityRaycaster(self.cfg)

        a_initial = self.raycaster.build_initial_actions(obs, self.theta_values)
        return a_initial
    




    def _refine_explored(self, a_initial_raw, agent_state, voxel):
        """
        Classify each (r, theta) in a_initial_raw as explored/unexplored by reading
        the provided voxel (no writes).

        Returns:
            list of (r, theta, is_unexplored)
        Notes:
            - Uses clip_frac and max_action_dist to pick the check point (same convention
            as VLM agent: use clipped distance for explored-check).
            - Uses the same small cross-shaped neighborhood and threshold (score < 3)
            as the original code to decide unexplored vs explored.
        """
        import numpy as np

        if not a_initial_raw:
            return []

        # Ensure voxel has voxels initialized (init_pos must be set externally by caller)
        # If voxel.init_pos is None, we cannot reliably map to grid — mark everything explored = False.
        if getattr(voxel, "init_pos", None) is None:
            return [(r, theta, False) for (r, theta) in a_initial_raw]

        # Build merged topdown map: voxels with explored color override base voxel map
        topdown_map = voxel.voxel_map.copy()
        try:
            mask = np.all(voxel.explored_map == voxel.explored_color, axis=-1)
            topdown_map[mask] = voxel.explored_color
        except Exception:
            # If shapes or channels mismatch, fall back to using voxel.voxel_map only.
            topdown_map = voxel.voxel_map.copy()

        h, w, _ = topdown_map.shape

        def _count_explored_neighborhood(gx, gy, radius=2):
            """Count how many explored-colored pixels appear in cross (vertical + horizontal) neighborhood."""
            cnt = 0
            # vertical slice
            x = int(gx)
            y0 = max(0, int(gy) - radius)
            y1 = min(h - 1, int(gy) + radius)
            if 0 <= x < w and y0 <= y1:
                col = topdown_map[y0:y1+1, x]
                cnt += int(np.all(col == voxel.explored_color, axis=-1).sum())
            # horizontal slice
            y = int(gy)
            x0 = max(0, int(gx) - radius)
            x1 = min(w - 1, int(gx) + radius)
            if 0 <= y < h and x0 <= x1:
                row = topdown_map[y, x0:x1+1]
                cnt += int(np.all(row == voxel.explored_color, axis=-1).sum())
            return cnt

        out = []
        for r, theta in a_initial_raw:
            # Clip the check distance using same convention as you use when you visualize explored (clip_frac)
            clipped_r = min(r * self.clip_frac, self.max_action_dist)

            # Local point in agent frame (x, y, z)
            local_pt = np.array([clipped_r * np.sin(theta), 0.0, -clipped_r * np.cos(theta)])

            # Convert to global then to grid coords; handle exceptions defensively
            try:
                global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
                gx, gy = voxel._global_to_grid(global_pt)
            except Exception:
                # If mapping fails for any reason, conservatively mark as explored (i.e., not unexplored)
                out.append((r, theta, False))
                continue

            # clamp grid coords
            gx = max(0, min(gx, w - 1))
            gy = max(0, min(gy, h - 1))

            score = _count_explored_neighborhood(gx, gy, radius=2)

            # same threshold as original implementation: unexplored if neighborhood has < 3 explored pixels
            is_unexplored = score < 3

            out.append((r, theta, is_unexplored))

        return out


    def _filter_unexplored_to_pairs(actions):
        """
        actions: list of either
        - (mag, theta)
        - (mag, theta, explored_flag)  # explored_flag: truthy means unexplored? see note
        Returns: list of (mag, theta) for unexplored rays (or all if no flag present)
        """
        if not actions:
            return []


        first = actions[0]
        if isinstance(first, (list, tuple)) and len(first) >= 3:

            pairs = [(mag, theta) for mag, theta, flag in actions if flag]
            return pairs
        else:
            return [(mag, theta) for mag, theta in actions]




    def _refine_actions(self, a_initial):
        """
        Port of `_action_proposer` from agent.py, but with:
        - voxel map scoring removed
        - cooldown fallback removed

        This version is tolerant to input containing either:
            - (mag, theta)
            - (mag, theta, flag)

        If an element has a third value (flag), we will FILTER OUT any
        element whose flag is not False (i.e. we keep only entries with flag == False).
        Change the filtering logic below if you want the opposite behavior.

        Input:
            a_initial: list of (mag, theta)  OR list of (mag, theta, flag)

        Output:
            list of [mag, theta, is_unexplored]
        """
        if not a_initial:
            return []

        # --- Normalise / optionally filter incoming items that have a third flag ---
        normalized_pairs = []
        for item in a_initial:
            # allow lists/tuples of len 2 or >=3
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                mag, theta, flag = item[0], item[1], item[2]
                if flag is False:
                    normalized_pairs.append((mag, theta))
            else:
                mag, theta = item[0], item[1]
                normalized_pairs.append((mag, theta))

        # If nothing left after filtering, return empty
        if not normalized_pairs:
            return []
        

        

        # From here on, behave exactly like original implementation but use normalized_pairs
        min_angle = self.hard_spacing
        explore_bias = self.explore_bias
        clip_frac = self.clip_frac

        explore = explore_bias > 0

        # Group by theta and keep the minimum radius per angle
        unique = {}
        for mag, theta in normalized_pairs:
            unique.setdefault(theta, []).append(mag)

        arrowData = []

        # We just mark everything as "unexplored" (True).
        for theta, mags in unique.items():
            mag = min(mags)
            arrowData.append([clip_frac * mag, theta, True])  # [r', theta, is_unexplored]

        arrowData.sort(key=lambda x: x[1])  # sort by angle

        thetas = set()
        out = []

        # Filter by minimum clipped radius
        filtered = [x for x in arrowData if x[0] > self.min_action_dist]

        if not filtered:
            return []

        if explore:
            # 1) Take only "unexplored" actions
            f = [x for x in filtered if x[2]]
            if f:
                # Longest unexplored ray
                longest = max(f, key=lambda x: x[0])
                longest_theta = longest[1]
                longest_ndx = f.index(longest)

                out.append(
                    [longest[0], longest[1], longest[2]]
                )
                thetas.add(longest_theta)

                # Sweep forward
                for i in range(longest_ndx + 1, len(f)):
                    if all(abs(f[i][1] - t) > (min_angle * 1.0) for t in thetas):
                        out.append(
                            [f[i][0], f[i][1], f[i][2]]
                        )
                        thetas.add(f[i][1])

                # Sweep backward
                for i in range(longest_ndx - 1, -1, -1):
                    if all(abs(f[i][1] - t) > (min_angle * 1.0) for t in thetas):
                        out.append(
                            [f[i][0], f[i][1], f[i][2]]
                        )
                        thetas.add(f[i][1])

                # 2) Optionally add more rays where spacing allows (scaled by explore_bias)
                for r_i, theta_i, e_i in filtered:
                    if theta_i not in thetas and min(
                        abs(theta_i - t) for t in thetas
                    ) > min_angle * explore_bias:
                        out.append(
                            [r_i, theta_i, e_i]
                        )
                        thetas.add(theta_i)
        else:
            # No explore bias – just keep filtered rays
            out = filtered

        out.sort(key=lambda x: x[1])

        # Restore original (unclipped) distances before returning,
        # and filter out rays whose true length < min_action_dist.
        original_distance_dict = {}
        for mag, theta, _ in a_initial:
            # If a theta appears multiple times, keep the *min* distance
            if theta not in original_distance_dict:
                original_distance_dict[theta] = mag
            else:
                original_distance_dict[theta] = min(original_distance_dict[theta], mag)

        max_dist = self.max_action_dist

        out = [
            [min(original_distance_dict.get(theta, mag), max_dist), theta, e]
            for mag, theta, e in out
        ]

        return out


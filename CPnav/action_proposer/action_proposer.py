# CPnav/action_proposer/action_proposer.py

import numpy as np
from .navigability_raycast import DepthNavigabilityRaycaster


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
        self.observe_frac = ap_cfg["observe_frac"]

        # Explore bias
        if ap_cfg.get("enable", False):
            self.explore_bias = ap_cfg.get("weight", 1.0)
        else:
            self.explore_bias = 0.0

        # Angular range / sensor range (use camera FOV)
        sensor_fov_deg = sim_cfg["sensor_cfg"]["fov"]
        half_range_rad = np.deg2rad(sensor_fov_deg)*2 / 2.0

        # Precompute theta samples in the camera FOV
        self.theta_values = np.linspace(-half_range_rad, half_range_rad, self.num_theta)

        # Depth-based raycaster helper: does RGBD → (r, theta)
        # Assumes DepthNavigabilityRaycaster(cfg) is how we constructed it.

    # ---------- public API ----------

    def propose(self, obs: dict, voxel_map=None):
        """
        Full pipeline:

          1) a_initial = _build_initial_actions(obs)
          2) a_final   = _refine_actions(a_initial)
          3) (optional) voxel_map update – NOT used yet.

        Args
        ----
        obs : dict
            CPNav observation. Must at least contain 'agent_state',
            'color_sensor', 'depth_sensor'.

        voxel_map : optional
            Currently ignored. Hook is left here for future integration.

        Returns
        -------
        list[tuple[float, float]]
            List of (r, theta) in the agent frame.
        """
        # 1) initial actions from depth (ray casting)
        a_initial = self._build_initial_actions(obs)

        # 2) refine according to spacing / explore bias
        a_final = self._refine_actions(a_initial)

        # 3) voxel_map update would go here in the future if you want it.

        # Drop the "is_unexplored" flag; return just (r, theta)
        return [(r, theta) for (r, theta, _) in a_final]

    # ---------- internal helpers ----------

    def _build_initial_actions(self, obs: dict):
        """
        Use DepthNavigabilityRaycaster to turn the current depth image into
        lidar-like radial distances:

            a_initial = [(r_i, theta_i), ...]

        where r_i is how far you can move along theta_i before hitting an obstacle.
        """
        # a_initial = [(self.max_action_dist, theta) for theta in self.theta_values]
        


        # print("sadasdasdsadasdasdasdasdassda",a_initial)
        # return a_initial
    

        self.raycaster = DepthNavigabilityRaycaster(self.cfg)

        a_initial = self.raycaster.build_initial_actions(obs, self.theta_values)
        return a_initial

    def _refine_actions(self, a_initial):
        """
        Port of `_action_proposer` from agent.py, but with:
        - voxel map scoring removed
        - cooldown fallback removed

        Input:
            a_initial: list of (mag, theta)

        Output:
            list of [mag, theta, is_unexplored]
        """
        if not a_initial:
            return []

        min_angle = self.hard_spacing
        explore_bias = self.explore_bias
        clip_frac = self.clip_frac
        observe_frac = self.observe_frac

        explore = explore_bias > 0

        # Group by theta and keep the minimum radius per angle
        unique = {}
        for mag, theta in a_initial:
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
                    [longest[0] * observe_frac, longest[1], longest[2]]
                )
                thetas.add(longest_theta)

                # Sweep forward
                for i in range(longest_ndx + 1, len(f)):
                    if all(abs(f[i][1] - t) > (min_angle * 1.0) for t in thetas):
                        out.append(
                            [f[i][0] * observe_frac, f[i][1], f[i][2]]
                        )
                        thetas.add(f[i][1])

                # Sweep backward
                for i in range(longest_ndx - 1, -1, -1):
                    if all(abs(f[i][1] - t) > (min_angle * 1.0) for t in thetas):
                        out.append(
                            [f[i][0] * observe_frac, f[i][1], f[i][2]]
                        )
                        thetas.add(f[i][1])

                # 2) Optionally add more rays where spacing allows (scaled by explore_bias)
                for r_i, theta_i, e_i in filtered:
                    if theta_i not in thetas and min(
                        abs(theta_i - t) for t in thetas
                    ) > min_angle * explore_bias:
                        out.append(
                            [r_i * observe_frac, theta_i, e_i]
                        )
                        thetas.add(theta_i)
        else:
            # No explore bias – just keep filtered rays
            out = filtered

        out.sort(key=lambda x: x[1])

        # Restore original (unclipped) distances before returning,
        # and filter out rays whose true length < min_action_dist.
        original_distance_dict = {}
        for mag, theta in a_initial:
            # If a theta appears multiple times, keep the *min* distance
            if theta not in original_distance_dict:
                original_distance_dict[theta] = mag
            else:
                original_distance_dict[theta] = min(original_distance_dict[theta], mag)

        min_dist = self.min_action_dist
        out = [
            [original_distance_dict.get(theta, mag), theta, e]
            for mag, theta, e in out
            if original_distance_dict.get(theta, mag) >= min_dist
        ]

        return out


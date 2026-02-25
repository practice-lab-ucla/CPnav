from __future__ import annotations

import numpy as np
try:
    import habitat_sim
except ImportError:
    habitat_sim = None
import cv2
from typing import Tuple, Optional, List, Dict, Any

from ..utils import (
    agent_frame_to_image_coords,
    unproject_2d,
    global_to_local,
    local_to_global,
    find_intersections,
    calculate_focal_length,
    depth_to_height,
)

class DepthNavigabilityRaycaster:
    """
    Helper to turn the current depth image into lidar-like radial distances.
    
    1. Generates a 'Navigability Mask' by projecting depth to height maps.
    2. Raycasts in image space to find how far the agent can walk in each direction.
    """

    def __init__(self, full_cfg: Dict[str, Any]):
        """
        Args:
            full_cfg: The entire loaded config dictionary.
        """
        self.cfg = full_cfg
        
        # 1. Unpack Sections
        self.agent_cfg = full_cfg["agent_cfg"]
        self.sim_cfg = full_cfg["sim_cfg"]
        self.sensor_cfg = self.sim_cfg["sensor_cfg"]

        # 2. Extract Base Resolution
        self.fov = float(self.sensor_cfg["fov"])
        H, W = self.sensor_cfg["resolution"]
        
        # Downsample resolution if needed (optimization)
        res_factor = int(self.sensor_cfg.get("res_factor", 1))
        self.H = H // res_factor
        self.W = W // res_factor
        self.resolution = (self.H, self.W)

        # 3. Camera Intrinsics Handling (Sim-to-Real capability)
        ap_cfg = self.agent_cfg.get("action_proposer_cfg", {})
        
        # Search for explicit parameters in either config section
        cfg_fx = ap_cfg.get("fx", ap_cfg.get("focal_length", self.sensor_cfg.get("fx", self.sensor_cfg.get("focal_length"))))
        cfg_fy = ap_cfg.get("fy", ap_cfg.get("focal_length", self.sensor_cfg.get("fy", self.sensor_cfg.get("focal_length"))))
        cfg_cx = ap_cfg.get("cx", self.sensor_cfg.get("cx"))
        cfg_cy = ap_cfg.get("cy", self.sensor_cfg.get("cy"))

        if cfg_fx is not None and cfg_fy is not None and cfg_cx is not None and cfg_cy is not None:
            # Scale intrinsics according to the resolution downsample factor
            self.fx = float(cfg_fx) / res_factor
            self.fy = float(cfg_fy) / res_factor
            self.cx = float(cfg_cx) / res_factor
            self.cy = float(cfg_cy) / res_factor
        else:
            # Fallback to ideal calculation for Sim
            ideal_focal_length = calculate_focal_length(self.fov, self.W)
            self.fx = ideal_focal_length
            self.fy = ideal_focal_length
            self.cx = self.W / 2.0
            self.cy = self.H / 2.0

        # 4. Navigability Thresholds
        self.height_thresh = float(self.agent_cfg.get("navigability_height_threshold", 0.2))
        self.agent_radius = float(self.agent_cfg.get("agent_radius", 0.1))
        self.clearance = float(self.agent_cfg.get("clearance_m", 0.05))
        self.depth_aware = self.agent_cfg.get("Depth_aware", False)
        self.nav_mode = self.agent_cfg.get("navigability_mode", "depth_sensor")

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    def build_initial_actions(self, 
                              obs: Dict[str, Any], 
                              theta_values: np.ndarray) -> List[Tuple[float, float]]:
        # ... (Implementation unchanged until the agent_frame_to_image_coords call) ...
        agent_state = obs["agent_state"]
        
        if "color_sensor" in agent_state.sensor_states:
            sensor_state = agent_state.sensor_states["color_sensor"]
        elif "depth_sensor" in agent_state.sensor_states:
             sensor_state = agent_state.sensor_states["depth_sensor"]
        else:
            sensor_state = agent_state

        depth_image: np.ndarray = obs["depth_sensor"]

        navigability_mask = self._get_navigability_mask(depth_image, agent_state, sensor_state)

        # Updated mapping arguments to use explicit intrinsics
        start_pxl = agent_frame_to_image_coords(
            np.array([0.0, 0.0, 0.0]),
            agent_state,
            sensor_state,
            self.fx, self.fy, self.cx, self.cy,
        )
        
        if start_pxl is None:
            return []

        a_initial = []
        for theta_i in theta_values:
            r_i, theta_out = self._get_radial_distance(
                start_pxl,
                float(theta_i),
                navigability_mask,
                agent_state,
                sensor_state,
                depth_image,
            )
            
            if r_i is not None:
                a_initial.append((r_i, float(theta_out)))

        return a_initial

    # --------------------------------------------------------------------- #
    # Internal Logic                                                        #
    # --------------------------------------------------------------------- #

    def _get_navigability_mask(self, depth_image, agent_state, sensor_state):
        # Updated mapping arguments to use explicit intrinsics
        height_map = depth_to_height(
            depth_image,
            self.fx, self.fy, self.cx, self.cy,
            sensor_state.position,
            sensor_state.rotation,
        )

        agent_height = float(agent_state.position[1])
        navigability_mask = np.abs(height_map - agent_height) < self.height_thresh

        return navigability_mask

    def _get_radial_distance(
        self,
        start_pxl: Tuple[int, int],
        theta_i: float,
        navigability_mask: np.ndarray,
        agent_state: habitat_sim.AgentState,
        sensor_state: habitat_sim.SixDOFPose,
        depth_image: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:
        
        H, W = self.resolution

        agent_point = np.array([2.0 * np.sin(theta_i), 0.0, -2.0 * np.cos(theta_i)])
        
        # Updated mapping arguments to use explicit intrinsics
        end_pxl = agent_frame_to_image_coords(
            agent_point,
            agent_state,
            sensor_state,
            self.fx, self.fy, self.cx, self.cy,
        )

        if end_pxl is None:
            return None, None

        intersections = find_intersections(
            int(start_pxl[0]), int(start_pxl[1]),
            int(end_pxl[0]), int(end_pxl[1]),
            W, H
        )
        if intersections is None:
            return None, None

        (x1, y1), (x2, y2) = intersections
        
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        x_coords = np.linspace(x1, x2, num_points).astype(int)
        y_coords = np.linspace(y1, y2, num_points).astype(int)

        if not navigability_mask[y_coords[0], x_coords[0]]:
            return 0.0, theta_i

        collision_idx = 0
        collision_found = False
        limit = max(1, num_points - 4) 

        for i in range(limit):
            collision_idx = i
            x, y = x_coords[i], y_coords[i]

            if self.depth_aware:
                z_here = float(depth_image[y, x]) if depth_image is not None else float("inf")
                
                if not np.isfinite(z_here) or z_here <= 0.0:
                    collision_found = True
                    break

                clearance_world = self.agent_radius + self.clearance
                # Width in pixels maps correctly to horizontal focal length
                half_w = int(np.ceil(clearance_world * self.fx / z_here))

                if half_w > 0:
                    x0, x1_w = max(0, x - half_w), min(W - 1, x + half_w)
                    y0, y1_h = max(0, y - half_w), min(H - 1, y + half_w)
                    
                    if not np.all(navigability_mask[y0 : y1_h+1, x0 : x1_w+1]):
                        collision_found = True
                        break

            end_idx = min(i + 4, num_points)
            lookahead_slice = navigability_mask[y_coords[i:end_idx], x_coords[i:end_idx]]

            if len(lookahead_slice) > 0 and np.sum(lookahead_slice) <= (len(lookahead_slice) // 2):
                collision_found = True
                break

        if collision_idx < 5:
            return 0.0, theta_i

        x_last, y_last = x_coords[collision_idx], y_coords[collision_idx]
        
        if self.nav_mode == "segmentation":
            r_i = 0.0794 * np.exp(0.006590 * collision_idx) + 0.616
        else:
            d = float(depth_image[y_last, x_last])
            
            # Updated mapping arguments to use explicit intrinsics
            camera_coords = unproject_2d(
                x_last, y_last, d,
                self.fx, self.fy, self.cx, self.cy,
            )
            
            world_coords = local_to_global(
                sensor_state.position,
                sensor_state.rotation,
                camera_coords,
            )
            
            local_coords = global_to_local(
                agent_state.position,
                agent_state.rotation,
                world_coords,
            )
            
            r_i = float(np.linalg.norm([local_coords[0], local_coords[2]]))

        return r_i, theta_i
import numpy as np
import habitat_sim
import cv2
from typing import Tuple, Optional, List, Dict, Any

from ..utils import (
    depth_to_height,
    agent_frame_to_image_coords,
    unproject_2d,
    global_to_local,
    local_to_global,
    find_intersections,
    calculate_focal_length,
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

        # 2. Camera Intrinsics
        self.fov = float(self.sensor_cfg["fov"])
        H, W = self.sensor_cfg["resolution"]
        
        # Downsample resolution if needed (optimization)
        res_factor = int(self.sensor_cfg.get("res_factor", 1))
        self.H = H // res_factor
        self.W = W // res_factor
        self.resolution = (self.H, self.W)

        self.focal_length = calculate_focal_length(self.fov, self.W)

        # 3. Navigability Thresholds (from Agent Config)
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
        """
        Compute available radial distances for a set of angles.

        Args:
            obs: Dictionary containing 'agent_state' and 'depth_sensor'.
            theta_values: Array of angles (radians) to test.

        Returns:
            List of (radius, theta) tuples.
        """
        agent_state: habitat_sim.AgentState = obs["agent_state"]
        
        # Ensure we have the sensor state (handle if it's missing, though unlikely in Sim)
        if "color_sensor" in agent_state.sensor_states:
            sensor_state = agent_state.sensor_states["color_sensor"]
        elif "depth_sensor" in agent_state.sensor_states:
             sensor_state = agent_state.sensor_states["depth_sensor"]
        else:
            # Fallback to agent body if sensor state is weirdly missing
            sensor_state = agent_state

        depth_image: np.ndarray = obs["depth_sensor"]

        # 1. Create Navigability Mask (Floor detection)
        navigability_mask = self._get_navigability_mask(depth_image, agent_state, sensor_state)

        # 2. Find Agent 'Start' Pixel (Bottom center projection)
        start_pxl = agent_frame_to_image_coords(
            np.array([0.0, 0.0, 0.0]),
            agent_state,
            sensor_state,
            resolution=self.resolution,
            focal_length=self.focal_length,
        )
        
        if start_pxl is None:
            return []

        # 3. Raycast for each angle
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
        """
        Generates a boolean mask where True = Walkable Floor.
        """
        # Convert depth image to world-height map
        height_map = depth_to_height(
            depth_image,
            self.fov,
            sensor_state.position,
            sensor_state.rotation,
        )

        agent_height = float(agent_state.position[1])
        
        # Pixel is navigable if its height is close to agent's floor height
        navigability_mask = np.abs(height_map - agent_height) < self.height_thresh

        # Optional: Save debug mask
        # mask_img = (navigability_mask.astype(np.uint8) * 255)
        # cv2.imwrite("navigability_mask.png", mask_img)

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
        """
        Traces a line in the image corresponding to direction theta_i.
        Stops when the line hits a non-navigable pixel.
        """
        H, W = self.resolution

        # 1. Calculate End Pixel (arbitrary large distance, e.g. 2m out)
        #    Habitat coords: -Z is forward, X is right.
        agent_point = np.array([2.0 * np.sin(theta_i), 0.0, -2.0 * np.cos(theta_i)])
        
        end_pxl = agent_frame_to_image_coords(
            agent_point,
            agent_state,
            sensor_state,
            resolution=self.resolution,
            focal_length=self.focal_length,
        )

        if end_pxl is None:
            return None, None

        # 2. Get Line Pixels
        intersections = find_intersections(
            int(start_pxl[0]), int(start_pxl[1]),
            int(end_pxl[0]), int(end_pxl[1]),
            W, H
        )
        if intersections is None:
            return None, None

        (x1, y1), (x2, y2) = intersections
        
        # Generate pixel coordinates along the line
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        x_coords = np.linspace(x1, x2, num_points).astype(int)
        y_coords = np.linspace(y1, y2, num_points).astype(int)

        # Check immediate start for collision
        if not navigability_mask[y_coords[0], x_coords[0]]:
            return 0.0, theta_i

        # 3. Walk the Line
        collision_idx = 0
        collision_found = False
        
        # We stop 4 pixels short to allow the "lookahead" check
        limit = max(1, num_points - 4) 

        for i in range(limit):
            collision_idx = i
            x, y = x_coords[i], y_coords[i]

            # A. Depth-Aware Clearance Check (Width of agent)
            if self.depth_aware:
                z_here = float(depth_image[y, x]) if depth_image is not None else float("inf")
                
                # Invalid depth = obstacle
                if not np.isfinite(z_here) or z_here <= 0.0:
                    collision_found = True
                    break

                # Calculate required pixel width at this depth
                # clearance_world = radius + safety_margin
                clearance_world = self.agent_radius + self.clearance
                half_w = int(np.ceil(clearance_world * self.focal_length / z_here))

                if half_w > 0:
                    x0, x1_w = max(0, x - half_w), min(W - 1, x + half_w)
                    y0, y1_h = max(0, y - half_w), min(H - 1, y + half_w)
                    
                    # If the box around center isn't fully navigable, we collide
                    # (Note: this is strict; requires full box to be white)
                    if not np.all(navigability_mask[y0 : y1_h+1, x0 : x1_w+1]):
                        collision_found = True
                        break

            # B. Lookahead Check (Noise reduction)
            # Slice the next 4 pixels (or fewer if near the end of the line)
            end_idx = min(i + 4, num_points)
            lookahead_slice = navigability_mask[y_coords[i:end_idx], x_coords[i:end_idx]]

            # If we are at the end or most of the next pixels are non-navigable, stop.
            if len(lookahead_slice) > 0 and np.sum(lookahead_slice) <= (len(lookahead_slice) // 2):
                collision_found = True
                break

        # If blocked immediately
        if collision_idx < 5:
            return 0.0, theta_i

        # 4. Convert Collision Pixel to Metric Distance
        # Last valid pixel
        x_last, y_last = x_coords[collision_idx], y_coords[collision_idx]
        
        if self.nav_mode == "segmentation":
            # Empirical heuristic (legacy)
            r_i = 0.0794 * np.exp(0.006590 * collision_idx) + 0.616
        else:
            # Geometric Unprojection
            d = float(depth_image[y_last, x_last])
            
            # 2D Image -> 3D Camera Frame
            camera_coords = unproject_2d(
                x_last, y_last, d,
                resolution=self.resolution,
                focal_length=self.focal_length,
            )
            
            # 3D Camera Frame -> 3D World Frame
            world_coords = local_to_global(
                sensor_state.position,
                sensor_state.rotation,
                camera_coords,
            )
            
            # 3D World Frame -> 3D Agent Frame
            local_coords = global_to_local(
                agent_state.position,
                agent_state.rotation,
                world_coords,
            )
            
            # Distance on ground plane (X, Z)
            r_i = float(np.linalg.norm([local_coords[0], local_coords[2]]))

        return r_i, theta_i
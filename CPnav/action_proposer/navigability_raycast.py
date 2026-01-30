# CPnav/action_proposer/navigability_raycast.py

import numpy as np
import habitat_sim
import cv2

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

    - Uses a depth-based navigability mask on the ground.
    - For each theta, traces a line in image space (like your old `_get_radial_distance`)
      and converts the collision pixel back to a metric distance r_i.
    """

    def __init__(self, full_cfg: dict):
        # full_cfg is the entire CPNav YAML dict.
        agent_cfg = full_cfg["agent_cfg"]
        sim_cfg = full_cfg["sim_cfg"]
        sensor_cfg = sim_cfg["sensor_cfg"]

        self.cfg = agent_cfg  # so we can use self.cfg['Depth_aware'], etc.

        # Camera intrinsics / geometry
        self.fov = float(sensor_cfg["fov"])  # hfov in degrees

        H, W = sensor_cfg["resolution"]
        res_factor = sensor_cfg.get("res_factor", 1)
        H //= res_factor
        W //= res_factor
        self.resolution = (H, W)

        self.focal_length = calculate_focal_length(self.fov, W)

        # Navigability parameters
        self.height_thresh = float(agent_cfg.get("navigability_height_threshold", 0.2))

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    def build_initial_actions(self, obs: dict, theta_values: np.ndarray):
        """
        Compute a_initial = [(r_i, theta_i), ...] for the current observation.

        Args:
            obs: dict with at least:
                 - 'agent_state' (habitat_sim.AgentState)
                 - 'depth_sensor' (H x W, depth in meters)
            theta_values: 1D array of angles in agent frame (radians).

        Returns:
            List[(r_i, theta_i)].
        """
        agent_state: habitat_sim.AgentState = obs["agent_state"]
        sensor_state: habitat_sim.SixDOFPose = agent_state.sensor_states["color_sensor"]
        depth_image: np.ndarray = obs["depth_sensor"]

        # 1) Navigability mask from depth
        navigability_mask = self._get_navigability_mask(depth_image, agent_state, sensor_state)

        # 2) Agent origin → image pixel
        start_pxl = agent_frame_to_image_coords(
            [0.0, 0.0, 0.0],
            agent_state,
            sensor_state,
            resolution=self.resolution,
            focal_length=self.focal_length,
        )
        if start_pxl is None:
            # Fallback: no rays
            return []

        a_initial = []
        H, W = navigability_mask.shape

        for theta_i in theta_values:
            r_i, theta_out = self._get_radial_distance(
                start_pxl,
                float(theta_i),
                navigability_mask,
                agent_state,
                sensor_state,
                depth_image,
            )
            if r_i is None:
                continue

            a_initial.append((r_i, float(theta_out)))

        return a_initial

    # --------------------------------------------------------------------- #
    # Internals                                                             #
    # --------------------------------------------------------------------- #

    def _get_navigability_mask(self, depth_image, agent_state, sensor_state):
        """
        Compute a boolean navigability mask from the depth image.

        Steps:
          - depth → global height map
          - compare each pixel's height to agent height
          - True if |height - agent_height| < height_thresh

        Also saves a debug image where:
          - True  → white (255)
          - False → black (0)
        """
        height_map = depth_to_height(
            depth_image,
            self.fov,  # degrees
            sensor_state.position,
            sensor_state.rotation,
        )

        agent_height = float(agent_state.position[1])
        target_height = agent_height  # you removed the -0.04 bias

        navigability_mask = np.abs(height_map - target_height) < self.height_thresh

        # --- Debug image: True = white, False = black ---
        mask_img = np.zeros_like(navigability_mask, dtype=np.uint8)
        mask_img[navigability_mask] = 255
        cv2.imwrite("navigability_mask.png", mask_img)

        return navigability_mask

    def _get_radial_distance(
        self,
        start_pxl: tuple,
        theta_i: float,
        navigability_mask: np.ndarray,
        agent_state: habitat_sim.AgentState,
        sensor_state: habitat_sim.SixDOFPose,
        depth_image: np.ndarray,
    ):
        """
        Your original ray-cast style function.

        Calculates the distance r_i that the agent can move in the
        direction theta_i, according to the navigability mask.
        """
        H, W = navigability_mask.shape

        # A point 2m out along theta in the agent frame → image pixel
        agent_point = [2 * np.sin(theta_i), 0.0, -2 * np.cos(theta_i)]
        end_pxl = agent_frame_to_image_coords(
            agent_point,
            agent_state,
            sensor_state,
            resolution=self.resolution,
            focal_length=self.focal_length,
        )
        if end_pxl is None or end_pxl[1] >= self.resolution[0]:
            return None, None

        # Find intersections of the theoretical line with the image boundaries
        intersections = find_intersections(
            int(start_pxl[0]),
            int(start_pxl[1]),
            int(end_pxl[0]),
            int(end_pxl[1]),
            W,
            H,
        )
        if intersections is None:
            return None, None

        (x1, y1), (x2, y2) = intersections
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        x_coords = np.linspace(x1, x2, num_points)
        y_coords = np.linspace(y1, y2, num_points)

        out = (int(x_coords[-1]), int(y_coords[-1]))
        if not navigability_mask[int(y_coords[0]), int(x_coords[0])]:
            return 0.0, theta_i

        # March along pixels until they become non-navigable
        last_i = 0
        for i in range(num_points - 4):
            last_i = i
            y = int(y_coords[i])
            x = int(x_coords[i])

            # ---------- Depth-aware footprint clearance ----------
            if self.cfg.get("Depth_aware", False):
                # Current depth at (x, y). If depth is missing/invalid, treat as collision.
                z_here = float(depth_image[y, x]) if depth_image is not None else float("inf")
                if not np.isfinite(z_here) or z_here <= 0.0:
                    out = (x, y)
                    break

                # Required world-space clearance (agent body radius + extra safety)
                clearance_world = float(self.cfg["agent_radius"]) + float(self.cfg["clearance_m"])

                # Convert meters -> pixels using focal length and current depth
                half_w = int(np.ceil(clearance_world * self.focal_length / z_here))

                if half_w > 0:
                    # A conservative square window around the centerline pixel
                    x0, x1w = max(0, x - half_w), min(W - 1, x + half_w)
                    y0, y1h = max(0, y - half_w), min(H - 1, y + half_w)

                    # If ANY pixel in the window is not navigable, the footprint would collide
                    if not np.all(navigability_mask[y0 : y1h + 1, x0 : x1w + 1]):
                        out = (x, y)
                        break
            # ---------- end depth-aware stripe ----------

            # Look ahead 4 pixels; if ≤2 of them are navigable, treat as collision
            if sum(
                navigability_mask[int(y_coords[j]), int(x_coords[j])]
                for j in range(i, i + 4)
            ) <= 2:
                out = (x, y)
                break

        # If we stopped almost immediately, treat it as blocked.
        if last_i < 5:
            return 0.0, theta_i

        # Distance computation depends on mode
        if self.cfg["navigability_mode"] == "segmentation":
            # Simple empirical estimation based on number of pixels
            r_i = 0.0794 * np.exp(0.006590 * last_i) + 0.616
        else:
            # Use depth to get metric distance
            out = (np.clip(out[0], 0, W - 1), np.clip(out[1], 0, H - 1))
            d = float(depth_image[out[1], out[0]])
            camera_coords = unproject_2d(
                out[0],
                out[1],
                d,
                resolution=self.resolution,
                focal_length=self.focal_length,
            )
            local_coords = global_to_local(
                agent_state.position,
                agent_state.rotation,
                local_to_global(
                    sensor_state.position,
                    sensor_state.rotation,
                    camera_coords,
                ),
            )
            r_i = float(np.linalg.norm([local_coords[0], local_coords[2]]))

        return r_i, theta_i

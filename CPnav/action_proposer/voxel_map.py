# CPnav/action_proposer/voxel_map.py

import numpy as np
import cv2
from ..utils import GREY, GREEN, local_to_global  # adjust imports as you have them

class VoxelMap:
    def __init__(self, cfg):
        self.cfg = cfg["agent_cfg"]

        # Same constants as old VLMNavAgent
        self.explored_color = GREY
        self.unexplored_color = GREEN
        self.map_size = self.cfg.get("map_size", 5000)
        self.voxel_ray_size = self.cfg.get("voxel_ray_size", 60)

        # Big maps
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros_like(self.voxel_map)

        # You likely already track this from your original code:
        self.init_pos = None   # set this once from the first agent_state
        self.scale = self.cfg.get("map_scale", 50.0)  # meters → pixels

    # ---- helpers copied from old Agent ----
    def _global_to_grid(self, position, rotation=None):
        dx = position[0] - self.init_pos[0]
        dz = position[2] - self.init_pos[2]
        h, w, _ = self.voxel_map.shape
        x = int(w // 2 + dx * self.scale)
        y = int(h // 2 + dz * self.scale)

        if rotation is not None:
            original = np.array([x, y, 1])
            new = rotation @ original
            return int(new[0]), int(new[1])
        return x, y

    def _update_voxel_single(self, r, theta, agent_state, clip_dist, clip_frac):
        agent_coords = self._global_to_grid(agent_state.position)

        # --- unexplored (green) ---
        unclipped = max(r, 0)
        local_pt = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
        p = self._global_to_grid(global_pt)
        cv2.line(self.voxel_map, agent_coords, p, self.unexplored_color, self.voxel_ray_size)

        # --- explored (grey) ---
        clipped = min(clip_frac * r, clip_dist)


        local_pt = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
        p = self._global_to_grid(global_pt)

        cv2.line(self.explored_map, agent_coords, p, self.explored_color, self.voxel_ray_size)

    # ---- public wrappers we will call from test_load.py ----

    def update_from_actions(self, a_initial_raw, agent_state, max_action_dist, clip_frac):
        """
        Update voxel_map / explored_map from the raw rays.
        a_initial_raw: list of (r, theta); r will be clipped to max_action_dist.
        """
        if self.init_pos is None:
            # set the origin on first use
            self.init_pos = np.array(agent_state.position, dtype=float)

        for r, theta in a_initial_raw:
            self._update_voxel_single(r, float(theta), agent_state,
                clip_dist=max_action_dist, clip_frac=clip_frac
            )

    def generate_voxel_image(self, projected_mapping, agent_state, chosen_action=None):
        """
        projected_mapping: {(r, theta) -> int label} coming from project_actions_on_image.
        Returns ONE top-down image with explored + unexplored + arrows+numbers.
        """
        if self.init_pos is None:
            self.init_pos = np.array(agent_state.position, dtype=float)

        agent_coords = self._global_to_grid(agent_state.position)

        right = (agent_state.position[0]+9, 0, agent_state.position[2])
        right_coords = self._global_to_grid(np.array(right))
        delta = abs(agent_coords[0] - right_coords[0])

        # Merge explored onto base map
        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        # Draw actions with numbers that match RGB
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = 1.25
        text_thickness = 1

        for (r, theta), action_id in projected_mapping.items():
            local_pt = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
            global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
            act_coords = self._global_to_grid(global_pt)

            # agent_coords = [agent_coords[0]+25, agent_coords[1]]

            cv2.arrowedLine(topdown_map, tuple(agent_coords), tuple(act_coords),
                            (0, 0, 255), 5, tipLength=0.05)
            text = str(action_id)
            (tw, th), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            center = (act_coords[0], act_coords[1])
            radius = max(tw, th) // 2 + 15

            color_circle = (0, 255, 0) if (chosen_action is not None and action_id == chosen_action) else (255, 255, 255)
            cv2.circle(topdown_map, center, radius, color_circle, -1)
            cv2.circle(topdown_map, center, radius, (0, 0, 255), 1)

            text_pos = (center[0] - tw // 2, center[1] + th // 2)
            cv2.putText(topdown_map, text, text_pos, font, text_size, (0, 0, 255), text_thickness + 1)

        # draw agent
        cv2.circle(topdown_map, agent_coords, radius=15, color=(0, 0, 255), thickness=-1)

        # simple zoom around the agent
        h, w, _ = topdown_map.shape
        x, y = agent_coords
        x1 = max(0, x - delta)
        x2 = min(w, x + delta)
        y1 = max(0, y - delta)
        y2 = min(h, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]
        return zoomed_map

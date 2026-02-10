import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional

# Assuming these exist in your utils, otherwise defining defaults
from ..utils import local_to_global

# Define colors if not imported from utils (BGR format for OpenCV)
GREY = (128, 128, 128)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class VoxelMap:
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initializes the Top-Down Map (Voxel Map).
        
        Args:
            cfg: The full dictionary from the new config file.
        """
        self.agent_cfg = cfg["agent_cfg"]

        # 1. Map Dimensions & Scale
        self.map_size = 5000  # Large canvas size
        self.scale = float(self.agent_cfg.get("map_scale", 100.0)) # pixels per meter
        
        # 2. Ray/Line Thickness
        # Calculate thickness based on agent diameter (2 * radius) in pixels
        # If agent_radius is 0.1m and scale is 100, radius is 10px, diameter is 20px.
        agent_radius = float(self.agent_cfg.get("agent_radius", 0.1))
        self.voxel_ray_size = int((agent_radius * 2 * self.scale))
        # Ensure a minimum visibility thickness
        self.voxel_ray_size = max(self.voxel_ray_size, 5)

        # 3. Colors
        self.explored_color = GREY
        self.unexplored_color = GREEN

        # 4. Initialize Maps
        # explored_map: tracks where the agent has been (clipped distance)
        # voxel_map: tracks what the agent has seen (unclipped raycast)
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros_like(self.voxel_map)

        self.init_pos = None  # Will be set on first update

    def _global_to_grid(self, position: np.ndarray) -> Tuple[int, int]:
        """
        Converts global world coordinates (x, y, z) to pixel coordinates (row, col).
        Uses self.init_pos as the center reference.
        """
        if self.init_pos is None:
            # Fallback if accessed before first update (shouldn't happen)
            return self.map_size // 2, self.map_size // 2

        # Calculate offset from start position
        dx = position[0] - self.init_pos[0]
        # Habitat uses -Z as forward usually, but map logic is standard 2D
        dz = position[2] - self.init_pos[2]

        # Center in the map
        h, w, _ = self.voxel_map.shape
        x = int(w // 2 + dx * self.scale)
        # Invert Z for image Y axis? Usually Z increases "down" in top-down view logic 
        # matching image row increase.
        y = int(h // 2 + dz * self.scale) 

        return x, y

    def _update_voxel_single(self, 
                             r: float, 
                             theta: float, 
                             agent_state, 
                             clip_dist: float, 
                             clip_frac: float):
        """
        Draws a single ray onto the map.
        """
        agent_coords = self._global_to_grid(agent_state.position)

        # --- 1. Unexplored (Seen Free Space) ---
        # Draw the full length of the ray (Green)
        unclipped_r = max(r, 0)
        local_pt_unclipped = np.array([
            unclipped_r * np.sin(theta), 
            0, 
            -unclipped_r * np.cos(theta)
        ])
        global_pt_unclipped = local_to_global(agent_state.position, agent_state.rotation, local_pt_unclipped)
        
        p_unclipped = self._global_to_grid(global_pt_unclipped)
        
        cv2.line(self.voxel_map, agent_coords, p_unclipped, 
                 self.unexplored_color, self.voxel_ray_size)

        # --- 2. Explored (Traversable Space) ---
        # Draw the safe/clipped length (Grey)
        # This represents area the agent is "confident" is explored/safe.
        clipped_r = min(clip_frac * r, clip_dist)
        
        local_pt_clipped = np.array([
            clipped_r * np.sin(theta), 
            0, 
            -clipped_r * np.cos(theta)
        ])
        global_pt_clipped = local_to_global(agent_state.position, agent_state.rotation, local_pt_clipped)
        
        p_clipped = self._global_to_grid(global_pt_clipped)

        cv2.line(self.explored_map, agent_coords, p_clipped, 
                 self.explored_color, self.voxel_ray_size)

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    def update_from_actions(self, 
                            a_initial_raw: List[Tuple[float, float]], 
                            agent_state, 
                            max_action_dist: float, 
                            clip_frac: float):
        """
        Updates the internal map buffers based on new raycast data.
        
        Args:
            a_initial_raw: List of (r, theta) from the raycaster.
            agent_state: Current habitat agent state.
            max_action_dist: Maximum distance to mark as explored.
            clip_frac: Fraction of ray to mark as explored.
        """
        # Initialize origin on first frame
        if self.init_pos is None:
            self.init_pos = np.array(agent_state.position, dtype=float)

        # Draw every ray
        for r, theta in a_initial_raw:
            self._update_voxel_single(
                r, 
                float(theta), 
                agent_state,
                clip_dist=max_action_dist, 
                clip_frac=clip_frac
            )

    def generate_voxel_image(self, 
                             projected_mapping: Dict[Tuple[float, float], int], 
                             agent_state, 
                             chosen_action: Optional[int] = None) -> np.ndarray:
        """
        Generates the final Top-Down visualization image.
        
        Args:
            projected_mapping: Map of {(r, theta) -> action_id} for arrows.
            agent_state: Current agent state.
            chosen_action: The ID of the action the agent actually picked (for highlighting).
        
        Returns:
            np.ndarray: Cropped image centered on the agent.
        """
        if self.init_pos is None:
             self.init_pos = np.array(agent_state.position, dtype=float)

        # 1. Merge Maps
        # Start with the "Unexplored" (Green) map
        topdown_map = self.voxel_map.copy()
        
        # Overlay the "Explored" (Grey) map
        # Logic: Where explored_map is grey, overwrite topdown_map
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        # 2. Draw Actions (Arrows & Numbers)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = 1.0
        text_thickness = 2
        
        agent_coords = self._global_to_grid(agent_state.position)

        for (r, theta), action_id in projected_mapping.items():
            # Calculate arrow tip
            local_pt = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
            global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
            act_coords = self._global_to_grid(global_pt)

            # Draw Arrow
            cv2.arrowedLine(topdown_map, agent_coords, act_coords,
                            RED, thickness=5, tipLength=0.1)
            
            # Draw Circle & Text at tip
            text = str(action_id)
            (tw, th), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            
            center = (act_coords[0], act_coords[1])
            radius = max(tw, th) // 2 + 10

            # Highlight chosen action in Green, others White
            circle_color = GREEN if (chosen_action is not None and action_id == chosen_action) else WHITE
            
            cv2.circle(topdown_map, center, radius, circle_color, -1)
            cv2.circle(topdown_map, center, radius, RED, 1)

            text_pos = (center[0] - tw // 2, center[1] + th // 2)
            cv2.putText(topdown_map, text, text_pos, font, text_size, RED, text_thickness)

        # 3. Draw Agent (Red dot)
        cv2.circle(topdown_map, agent_coords, radius=10, color=RED, thickness=-1)

        # 4. Zoom / Crop
        # Crop a window of X meters around the agent
        view_window_m = 9.0 
        delta_px = int(view_window_m * self.scale)

        h, w, _ = topdown_map.shape
        x, y = agent_coords
        
        x1 = max(0, x - delta_px)
        x2 = min(w, x + delta_px)
        y1 = max(0, y - delta_px)
        y2 = min(h, y + delta_px)

        zoomed_map = topdown_map[y1:y2, x1:x2]
        
        return zoomed_map
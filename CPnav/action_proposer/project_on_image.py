import math
import cv2
import numpy as np
from typing import Iterable, Tuple, Dict, Optional, Any

from ..utils import (
    agent_frame_to_image_coords,
    calculate_focal_length,
    put_text_on_image,
)

# Constants for colors (BGR)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)


def _compute_intrinsics(full_cfg: Dict[str, Any]) -> Tuple[Tuple[int, int], float]:
    """
    Extracts resolution and focal length from the clean config structure.
    
    Path: full_cfg -> sim_cfg -> sensor_cfg
    """
    sim_cfg = full_cfg["sim_cfg"]
    sensor_cfg = sim_cfg["sensor_cfg"]
    
    H, W = sensor_cfg["resolution"]
    res_factor = int(sensor_cfg.get("res_factor", 1))

    # Apply downsampling factor
    H = H // res_factor
    W = W // res_factor

    fov = float(sensor_cfg["fov"])
    focal_length = calculate_focal_length(fov, W)

    return (H, W), focal_length


def _can_project_single(
    r_i: float,
    theta_i: float,
    agent_state,
    sensor_state,
    resolution: Tuple[int, int],
    focal_length: float,
    edge_threshold: float,
) -> Optional[Tuple[int, int]]:
    """
    Checks if a polar action (r, theta) projects to a valid pixel on screen,
    respecting edge borders.
    """
    # 3D Point in Agent Frame (Forward is -Z, Right is +X)
    agent_point = np.array([r_i * np.sin(theta_i), 0.0, -r_i * np.cos(theta_i)])

    end_px = agent_frame_to_image_coords(
        agent_point,
        agent_state,
        sensor_state,
        resolution=resolution,
        focal_length=focal_length,
    )

    if end_px is None:
        return None

    h, w = resolution
    x, y = end_px

    # Check bounds with edge threshold
    min_w, max_w = edge_threshold * w, (1.0 - edge_threshold) * w
    min_h, max_h = edge_threshold * h, (1.0 - edge_threshold) * h

    if min_w <= x <= max_w and min_h <= y <= max_h:
        return int(x), int(y)

    return None


def project_actions_on_image(
    cfg: Dict[str, Any],
    rgb_image: np.ndarray,
    actions: Iterable[Tuple[float, float]],
    agent_state,
    sensor_state,
    chosen_index: Optional[int] = None,
    initial_actions: Optional[Iterable[Tuple[float, float]]] = None,
    show_initial: bool = True,
    draw_turn_around: bool = True
) -> Dict[Tuple[float, float], int]:
    """
    Draws navigation actions (arrows) onto the RGB image.
    
    Args:
        cfg: The full clean config dictionary.
        rgb_image: The image to draw on (modified in-place).
        actions: Refined list of (r, theta) to propose to the VLM.
        agent_state: Current simulator agent state.
        sensor_state: Current simulator sensor state.
        chosen_index: Index of the action to highlight (if any).
        initial_actions: Raw list of raycast hits (for debugging).
        show_initial: Whether to draw small dots for initial actions.
        draw_turn_around: Whether to draw the '0' turn-around bubble.
    """
    # 1. Image Format Handling
    # Ensure standard BGR (drop Alpha if present)
    if rgb_image.ndim == 3 and rgb_image.shape[2] == 4:
        rgb_image = rgb_image[:, :, :3]

    # Ensure writable uint8
    if rgb_image.dtype != np.uint8:
        # np.ascontiguousarray helps if slice is non-contiguous
        rgb_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)

    # 2. Setup Params
    resolution, focal_length = _compute_intrinsics(cfg)
    
    # Thresholds from agent_cfg
    agent_cfg = cfg["agent_cfg"]
    edge_threshold = float(agent_cfg.get("image_edge_threshold", 0.04))

    # Scale visuals relative to 1080p standard
    scale_factor = rgb_image.shape[0] / 1080.0
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    projected_map: Dict[Tuple[float, float], int] = {}

    # 3. Find Camera Origin (Bottom-Center projection)
    start_px_float = agent_frame_to_image_coords(
        np.array([0.0, 0.0, 0.0]),
        agent_state,
        sensor_state,
        resolution=resolution,
        focal_length=focal_length,
    )

    if start_px_float is None:
        put_text_on_image(
            rgb_image,
            "CAMERA ORIGIN OFF-SCREEN",
            text_color=GREEN,
            text_size=2.0 * scale_factor,
            location="center",
            text_thickness=math.ceil(2 * scale_factor),
            highlight=False,
        )
        return projected_map

    start_px = (int(start_px_float[0]), int(start_px_float[1]))

    # 4. (Optional) Draw Initial Raw Actions for Debugging
    if show_initial and initial_actions:
        small_radius = max(3, math.ceil(4 * scale_factor))
        font_scale_small = 0.7 * scale_factor
        thickness_small = math.ceil(2 * scale_factor)

        for i, (r_raw, theta_raw) in enumerate(initial_actions):
            px_raw = agent_frame_to_image_coords(
                np.array([r_raw * np.sin(theta_raw), 0.0, -r_raw * np.cos(theta_raw)]),
                agent_state,
                sensor_state,
                resolution=resolution,
                focal_length=focal_length,
            )
            
            if px_raw is not None:
                px_int = (int(px_raw[0]), int(px_raw[1]))
                
                # Draw small orange dot
                cv2.circle(rgb_image, px_int, small_radius, ORANGE, -1)
                
                # Optional label "i1", "i2"...
                # cv2.putText(rgb_image, f"i{i}", (px_int[0]+5, px_int[1]+5), 
                #             font, font_scale_small, BLACK, thickness_small)

    # 5. Draw Valid Refined Actions (Arrows)
    action_counter = 1
    
    for idx, (r_i, theta_i) in enumerate(actions):
        # Check projection validity
        end_px = _can_project_single(
            r_i, theta_i,
            agent_state, sensor_state,
            resolution, focal_length,
            edge_threshold
        )
        
        if end_px is None:
            continue

        # Record this action as successfully projected
        projected_map[(r_i, theta_i)] = action_counter
        
        # --- Drawing Styles ---
        # 1. Main Arrow Line
        cv2.arrowedLine(
            rgb_image,
            start_px,
            end_px,
            RED,
            thickness=math.ceil(5 * scale_factor),
            tipLength=0.0, # We draw a circle at the tip instead
        )

        # 2. Number Circle at Tip
        label_text = str(action_counter)
        text_size = 2.4 * scale_factor
        text_thickness = math.ceil(3 * scale_factor)

        (text_w, text_h), _ = cv2.getTextSize(label_text, font, text_size, text_thickness)
        
        circle_radius = max(text_w, text_h) // 2 + math.ceil(15 * scale_factor)
        
        # Determine Circle Color (Highlight if chosen)
        fill_color = WHITE
        if chosen_index is not None and idx == chosen_index:
            fill_color = GREEN
        
        # Draw filled circle + border
        cv2.circle(rgb_image, end_px, circle_radius, fill_color, -1)
        cv2.circle(rgb_image, end_px, circle_radius, RED, math.ceil(2 * scale_factor))

        # 3. Draw Number
        text_origin = (
            end_px[0] - text_w // 2,
            end_px[1] + text_h // 2,
        )
        cv2.putText(
            rgb_image,
            label_text,
            text_origin,
            font,
            text_size,
            BLACK,
            text_thickness,
        )
        
        action_counter += 1

    # 6. Handle Special Cases (No actions / Turn Around)
    if not projected_map and not draw_turn_around:
        put_text_on_image(
            rgb_image,
            "NO VALID ACTIONS",
            text_color=GREEN,
            text_size=3.0 * scale_factor,
            location="center",
            text_thickness=math.ceil(3 * scale_factor),
            highlight=False,
        )
        
    elif draw_turn_around:
        # Draw "0 - Turn Around" Bubble on the left side
        H_img, W_img = rgb_image.shape[:2]
        
        bubble_text = "0"
        text_size = 3.0 * scale_factor
        text_thickness = math.ceil(3 * scale_factor)
        (tw, th), _ = cv2.getTextSize(bubble_text, font, text_size, text_thickness)

        # Position: Left side, vertically centered
        center_x = math.ceil(0.06 * W_img) 
        center_y = math.ceil(H_img / 2)
        radius = max(tw, th) // 2 + math.ceil(15 * scale_factor)

        # Draw Bubble
        cv2.circle(rgb_image, (center_x, center_y), radius, WHITE, -1)
        cv2.circle(rgb_image, (center_x, center_y), radius, RED, math.ceil(2 * scale_factor))

        # Draw Number
        text_pos = (center_x - tw // 2, center_y + th // 2)
        cv2.putText(rgb_image, bubble_text, text_pos, font, text_size, BLACK, text_thickness)

        # Draw Label "TURN AROUND" underneath
        label_scale = 1.0 * scale_factor
        label_thick = math.ceil(2 * scale_factor)
        cv2.putText(
            rgb_image,
            "TURN AROUND",
            (center_x - radius, center_y + radius + math.ceil(40 * scale_factor)),
            font,
            label_scale,
            RED,
            label_thick,
        )

    return projected_map
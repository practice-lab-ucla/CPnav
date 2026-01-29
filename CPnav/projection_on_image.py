# CPnav/projection_on_image.py

import math
from typing import Iterable, Tuple, Dict, Optional

import cv2
import numpy as np

from .utils import (
    agent_frame_to_image_coords,
    calculate_focal_length,
    put_text_on_image,
    BLACK,
    WHITE,
    RED,
    GREEN,
)


def _compute_intrinsics(full_cfg: dict) -> Tuple[Tuple[int, int], float]:
    """
    Compute (H, W) and focal length exactly like DepthNavigabilityRaycaster.
    Requires:
        full_cfg["sim_cfg"]["sensor_cfg"]["resolution"] = [H, W]
        full_cfg["sim_cfg"]["sensor_cfg"]["fov"]        = float (deg)
        full_cfg["sim_cfg"]["sensor_cfg"].get("res_factor", 1)
    """
    sensor_cfg = full_cfg["sim_cfg"]["sensor_cfg"]
    H, W = sensor_cfg["resolution"]
    res_factor = sensor_cfg.get("res_factor", 1)

    H //= res_factor
    W //= res_factor

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
    Check whether a polar action (r_i, theta_i) can be projected inside the
    image (i.e. not too close to the borders). Returns the pixel if valid,
    otherwise None.
    """
    # Same convention as in agent._can_project: forward = -z, left = +x
    agent_point = [r_i * np.sin(theta_i), 0.0, -r_i * np.cos(theta_i)]

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

    if (
        edge_threshold * w <= x <= (1.0 - edge_threshold) * w
        and edge_threshold * h <= y <= (1.0 - edge_threshold) * h
    ):
        return int(x), int(y)

    return None


def project_actions_on_image(
    cfg: dict,
    rgb_image: np.ndarray,
    actions: Iterable[Tuple[float, float]],
    agent_state,
    sensor_state,
    chosen_index: Optional[int] = None,
) -> Dict[Tuple[float, float], int]:
    """
    Projects (r, theta) actions onto the RGB image, similar to VLMNavAgent._project_onto_image.

    Parameters
    ----------
    cfg : dict
        Full CPNav config (same dict you pass to ObjectNavEnv / ActionProposer).
    rgb_image : np.ndarray
        H x W x 3 (or 4) uint8 array. Modified in-place (arrows and numbers drawn).
    actions : iterable of (r, theta)
        The output of ActionProposer.propose(obs).
    agent_state :
        obs["agent_state"] from your environment.
    sensor_state :
        agent_state.sensor_states["color_sensor"].
    chosen_index : int, optional
        Index into `actions` (0-based) to highlight. If None, nothing is specially highlighted.

    Returns
    -------
    dict
        Mapping {(r, theta) -> action_number} for all actions that were actually projected.
    """
    # If we accidentally get RGBA, drop the alpha
    if rgb_image.ndim == 3 and rgb_image.shape[2] == 4:
        rgb_image = rgb_image[:, :, :3]

    if rgb_image.dtype != np.uint8:
        rgb_image[:] = np.clip(rgb_image, 0, 255).astype(np.uint8)

    # Intrinsics and edge threshold
    resolution, focal_length = _compute_intrinsics(cfg)
    agent_cfg = cfg["agent_cfg"]
    edge_threshold = float(agent_cfg.get("image_edge_threshold", 0.04))

    scale_factor = rgb_image.shape[0] / 1080.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = BLACK
    circle_color = WHITE
    projected: Dict[Tuple[float, float], int] = {}

    # Agent origin in image
    start_px = agent_frame_to_image_coords(
        [0.0, 0.0, 0.0],
        agent_state,
        sensor_state,
        resolution=resolution,
        focal_length=focal_length,
    )

    if start_px is None:
        put_text_on_image(
            rgb_image,
            "NO CAMERA ORIGIN IN VIEW",
            text_color=GREEN,
            text_size=3.0 * scale_factor,
            location="center",
            text_thickness=math.ceil(3 * scale_factor),
            highlight=False,
        )
        return projected

    start_px = (int(start_px[0]), int(start_px[1]))

    for idx, (r_i, theta_i) in enumerate(actions):
        # Check whether this action can be projected (inside image, not border)
        end_px = _can_project_single(
            float(r_i),
            float(theta_i),
            agent_state,
            sensor_state,
            resolution,
            focal_length,
            edge_threshold,
        )
        if end_px is None:
            # action cannot be projected → skip
            continue

        action_name = len(projected) + 1  # 1-based label for arrows
        projected[(float(r_i), float(theta_i))] = action_name

        text_size = 2.4 * scale_factor
        text_thickness = math.ceil(3 * scale_factor)

        # Main arrow (red, like before)
        cv2.arrowedLine(
            rgb_image,
            start_px,
            end_px,
            RED,
            math.ceil(5 * scale_factor),
            tipLength=0.0,
        )

        # Draw a numbered circle at the end of the arrow
        text = str(action_name)
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, text_size, text_thickness
        )

        circle_center = (end_px[0], end_px[1])
        circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)

        # Fill color: highlight chosen_index if requested (0-based in the original `actions` list)
        if chosen_index is not None and idx == chosen_index:
            cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
        else:
            cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)

        # Circle border
        cv2.circle(
            rgb_image,
            circle_center,
            circle_radius,
            RED,
            math.ceil(2 * scale_factor),
        )

        # Put number text
        text_position = (
            circle_center[0] - text_width // 2,
            circle_center[1] + text_height // 2,
        )
        cv2.putText(
            rgb_image,
            text,
            text_position,
            font,
            text_size,
            text_color,
            text_thickness,
        )

    if not projected:
        put_text_on_image(
            rgb_image,
            "NO VALID ACTIONS TO PROJECT",
            text_color=GREEN,
            text_size=3.0 * scale_factor,
            location="center",
            text_thickness=math.ceil(3 * scale_factor),
            highlight=False,
        )

    return projected

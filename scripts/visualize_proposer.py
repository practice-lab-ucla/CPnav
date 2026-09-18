"""
Visualize CPnav's action-candidate generation on the robot's first-person view.

For each step of a scripted rollout (no VLM / API key needed) this renders:
  1. Navigable floor        - green tint   (DepthNavigabilityRaycaster mask)
  2. Explored region        - blue tint    (VoxelMap explored_map back-projected
                                            into the view, using the map state the
                                            exploration filter actually consulted)
  3. Action candidates BEFORE the exploration trim:
       - orange arrows: removed by the exploration filter (endpoint explored)
       - yellow arrows: unexplored but dropped by angular-spacing refinement
       - grey dots:     raw rays too short to be safe (< min_action_dist)
     plus the FINAL proposed actions in the standard red numbered style.

A top-down voxel map is saved side-by-side for context.

Usage (from the CPnav repo root, inside the CPnav conda env):
    python scripts/visualize_proposer.py --config config/CPnav_5_nano_131.yaml \
        --episode 100 --steps 12 --out logs/proposer_viz
"""

import os
import sys
import math
import argparse

import cv2
import yaml
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from CPnav.env import ObjectNavEnv, PolarAction
from CPnav.utils import local_to_global, agent_frame_to_image_coords
from CPnav.action_proposer import VoxelMap, ActionProposer, project_actions_on_image
from CPnav.action_proposer.project_on_image import _compute_intrinsics

# Colors in RGB (obs["color_sensor"] is RGB; converted to BGR only at imwrite)
C_NAV = (0, 200, 0)        # navigable floor tint
C_NAV_EDGE = (0, 90, 0)    # navigable-region boundary (darker)
C_EXPLORED = (30, 90, 255)  # explored floor tint
C_EXPLORED_EDGE = (10, 35, 120)  # explored-region boundary (darker)
C_TRIM_EXPLORED = (255, 140, 0)  # candidate removed by exploration filter
C_TRIM_SPACING = (255, 230, 0)   # candidate removed by angular spacing
C_TOO_SHORT = (160, 160, 160)    # raw ray below min_action_dist


def scaled_intrinsics(raycaster, shape):
    """Raycaster intrinsics rescaled to an image of the given (H, W)."""
    h, w = shape[:2]
    sx, sy = w / raycaster.W, h / raycaster.H
    return raycaster.fx * sx, raycaster.fy * sy, raycaster.cx * sx, raycaster.cy * sy


def explored_mask_in_view(depth, raycaster, voxel, sensor_state):
    """Back-project every depth pixel to the voxel grid and sample explored_map.

    Uses the voxel state BEFORE it is updated with the current frame, i.e. the
    exact map the exploration filter consults in _check_explored_status.
    """
    if getattr(voxel, "init_pos", None) is None:
        return np.zeros(depth.shape[:2], dtype=bool)

    h_img, w_img = depth.shape[:2]
    fx, fy, cx, cy = scaled_intrinsics(raycaster, depth.shape)

    i_idx, j_idx = np.indices((h_img, w_img))
    x_local = (j_idx - cx) * depth / fx
    y_local = (i_idx - cy) * depth / fy
    local_points = np.stack((x_local, -y_local, -depth), axis=-1)
    global_points = local_to_global(
        sensor_state.position, sensor_state.rotation, local_points
    )

    h_map, w_map, _ = voxel.explored_map.shape
    gx = (w_map // 2 + (global_points[..., 0] - voxel.init_pos[0]) * voxel.scale).astype(int)
    gy = (h_map // 2 + (global_points[..., 2] - voxel.init_pos[2]) * voxel.scale).astype(int)

    valid = (gx >= 0) & (gx < w_map) & (gy >= 0) & (gy < h_map) & (depth > 0)
    explored_grid = np.all(
        voxel.explored_map == np.array(voxel.explored_color, dtype=np.uint8), axis=-1
    )
    mask = np.zeros((h_img, w_img), dtype=bool)
    mask[valid] = explored_grid[gy[valid], gx[valid]]
    return mask


def tint(image, mask, color, alpha=0.45):
    """Alpha-blend a solid color into image where mask is True."""
    overlay = image[mask].astype(np.float32)
    image[mask] = ((1 - alpha) * overlay + alpha * np.array(color, np.float32)).astype(np.uint8)


def region_boundary(mask, thickness=5):
    """Boolean mask of the region's boundary (morphological gradient)."""
    kernel = np.ones((thickness, thickness), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)


def draw_candidate_arrow(image, r, theta, agent_state, sensor_state, intr, color,
                         thickness=2, dot_only=False, tip=0.015):
    """Draw one polar candidate on the first-person view."""
    (h, w), fx, fy, cx, cy = intr
    start = agent_frame_to_image_coords(
        np.array([0.0, 0.0, 0.0]), agent_state, sensor_state, fx, fy, cx, cy
    )
    end = agent_frame_to_image_coords(
        np.array([r * np.sin(theta), 0.0, -r * np.cos(theta)]),
        agent_state, sensor_state, fx, fy, cx, cy,
    )
    if end is None:
        return
    end = (int(np.clip(end[0], 0, w - 1)), int(np.clip(end[1], 0, h - 1)))
    if dot_only or start is None:
        cv2.circle(image, end, 8, color, -1)
        return
    start = (int(start[0]), int(start[1]))
    cv2.arrowedLine(image, start, end, color, thickness, tipLength=tip)


def render_topdown(voxel, agent_state, a_checked, projected, kept_thetas, ap,
                   view_window_m=9.0, explored_map=None):
    """Top-down map in the SAME color scheme as the first-person overlay.

    Green (seen free space) comes from voxel.voxel_map -- call AFTER
    voxel.update_from_actions so it reflects the CURRENT time step.
    Blue (explored) uses `explored_map` -- pass a pre-update copy so it shows
    the PREVIOUS step, i.e. the map state the exploration filter consulted.

    Visualization only -- reads the VoxelMap buffers, never writes them.
    """
    if getattr(voxel, "init_pos", None) is None:
        voxel.init_pos = np.array(agent_state.position, dtype=float)
    if explored_map is None:
        explored_map = voxel.explored_map

    seen = np.all(voxel.voxel_map ==
                  np.array(voxel.unexplored_color, np.uint8), axis=-1)
    explored = np.all(explored_map ==
                      np.array(voxel.explored_color, np.uint8), axis=-1)

    canvas = np.zeros_like(voxel.voxel_map)
    canvas[seen] = C_NAV
    canvas[explored & seen] = C_EXPLORED
    canvas[region_boundary(seen)] = C_NAV_EDGE
    canvas[region_boundary(explored & seen)] = C_EXPLORED_EDGE

    def endpoint_px(r, theta):
        local_pt = np.array([r * np.sin(theta), 0.0, -r * np.cos(theta)])
        global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
        return voxel._global_to_grid(global_pt)

    agent_px = voxel._global_to_grid(agent_state.position)
    # trimmed candidates (orange/yellow/grey) intentionally omitted on the
    # top-down -- it shows regions + final proposed actions only

    # final proposed actions: red numbered, matching the first-person style
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (r, theta), action_id in projected.items():
        end = endpoint_px(r, theta)
        cv2.arrowedLine(canvas, agent_px, end, (255, 0, 0), 5, tipLength=0.1)
        text = str(action_id)
        (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
        radius = max(tw, th) // 2 + 10
        cv2.circle(canvas, end, radius, (255, 255, 255), -1)
        cv2.circle(canvas, end, radius, (255, 0, 0), 2)
        cv2.putText(canvas, text, (end[0] - tw // 2, end[1] + th // 2),
                    font, 1.0, (255, 0, 0), 2)

    cv2.circle(canvas, agent_px, 10, (255, 255, 255), -1)
    cv2.circle(canvas, agent_px, 10, (0, 0, 0), 2)

    # crop a window around the agent (same convention as generate_voxel_image)
    delta = int(view_window_m * voxel.scale)
    h, w, _ = canvas.shape
    x, y = agent_px
    return canvas[max(0, y - delta):min(h, y + delta),
                  max(0, x - delta):min(w, x + delta)]


def draw_legend(image):
    entries = [
        ("navigable floor", C_NAV),
        ("navigable boundary", C_NAV_EDGE),
        ("explored floor (voxel map)", C_EXPLORED),
        ("explored boundary", C_EXPLORED_EDGE),
        ("trimmed: leads to explored", C_TRIM_EXPLORED),
        ("trimmed: angular spacing", C_TRIM_SPACING),
        ("raw ray < min_action_dist", C_TOO_SHORT),
        ("final proposed action", (255, 0, 0)),
    ]
    x0, y0, dy = 15, 30, 34
    for i, (label, color) in enumerate(entries):
        y = y0 + i * dy
        cv2.rectangle(image, (x0, y - 18), (x0 + 26, y + 4), color, -1)
        cv2.putText(image, label, (x0 + 36, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/CPnav_5_nano_131.yaml")
    parser.add_argument("--episode", type=int, default=100)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--out", default="logs/proposer_viz")
    parser.add_argument("--height_thresh", type=float, default=None,
                        help="Visualization-only override of "
                             "navigability_height_threshold (config untouched)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.height_thresh is not None:
        cfg["agent_cfg"]["navigability_height_threshold"] = args.height_thresh
        print(f"[viz] overriding navigability_height_threshold = {args.height_thresh}")

    env = ObjectNavEnv(cfg)
    ap = ActionProposer(cfg)
    voxel = VoxelMap(cfg)

    goal = env.initialize_episode(args.episode)
    obs = env.reset()
    os.makedirs(args.out, exist_ok=True)
    print(f"episode {args.episode}: goal = {goal}, saving to {args.out}")

    for step in range(args.steps):
        agent_state = obs["agent_state"]
        sensor_state = agent_state.sensor_states.get(
            "color_sensor", agent_state.sensor_states.get("depth_sensor", agent_state)
        )
        rgb = np.ascontiguousarray(obs["color_sensor"][:, :, :3], dtype=np.uint8)
        depth = np.squeeze(obs["depth_sensor"]).astype(np.float32)

        # --- Run the proposer stages exactly as propose_actions_and_update_voxel does,
        #     but keep the intermediates and delay the voxel update until after we
        #     visualize the map state the filter consulted.
        a_initial = ap._build_initial_actions(obs)
        a_checked = ap._check_explored_status(a_initial, agent_state, voxel)
        a_final = ap._refine_actions(a_checked)
        if not a_final and ap.propose_if_all_explored:
            a_final = ap._refine_actions([(r, t, True) for (r, t, _) in a_checked])
        final_actions = [(r, t) for (r, t, _) in a_final]
        kept_thetas = {t for (_, t) in final_actions}

        # --- 1 & 2: floor overlays (voxel state BEFORE this frame's update)
        nav_small = ap.raycaster._get_navigability_mask(depth, agent_state, sensor_state)
        explored_small = explored_mask_in_view(depth, ap.raycaster, voxel, sensor_state)
        h, w = rgb.shape[:2]
        nav = cv2.resize(nav_small.astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
        explored = cv2.resize(explored_small.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        view = rgb.copy()
        tint(view, nav & ~explored, C_NAV)
        tint(view, nav & explored, C_EXPLORED)
        # darker solid boundaries on top of the tints
        view[region_boundary(nav)] = C_NAV_EDGE
        view[region_boundary(nav & explored)] = C_EXPLORED_EDGE

        # --- 3: candidates before the exploration trim
        (res_hw), fx, fy, cx, cy = _compute_intrinsics(cfg)
        intr = ((h, w), fx, fy, cx, cy)
        n_expl = n_spac = 0  # draw every 2nd trimmed arrow (red finals untouched)
        for r, theta, is_unexplored in a_checked:
            draw_r = min(r, ap.max_action_dist)
            if r * ap.clip_frac <= ap.min_action_dist:
                pass  # too-short rays: not drawn
            elif not is_unexplored:
                n_expl += 1
                if n_expl % 2:
                    draw_candidate_arrow(view, draw_r, theta, agent_state, sensor_state,
                                         intr, C_TRIM_EXPLORED)
            elif theta not in kept_thetas:
                n_spac += 1
                if n_spac % 2:
                    draw_candidate_arrow(view, draw_r, theta, agent_state, sensor_state,
                                         intr, C_TRIM_SPACING)

        # final actions in the standard numbered style, on top
        # (skip the call when empty so it can't draw the "NO VALID ACTIONS" text)
        projected = {}
        if final_actions:
            projected = project_actions_on_image(
                cfg, view, final_actions, agent_state, sensor_state,
                chosen_index=None, show_initial=False, draw_turn_around=False,
            )

        # --- top-down panel: green (seen free space) = CURRENT step, so update
        #     the voxel map first (same order as the real pipeline); blue
        #     (explored) = PREVIOUS step, via a pre-update snapshot
        explored_prev = voxel.explored_map.copy()
        voxel.update_from_actions(a_initial, agent_state,
                                  max_action_dist=ap.max_action_dist,
                                  clip_frac=ap.clip_frac)
        topdown = render_topdown(voxel, agent_state, a_checked, projected,
                                 kept_thetas, ap, explored_map=explored_prev)
        if topdown.size:
            scale = h / topdown.shape[0]
            topdown = cv2.resize(topdown, (int(topdown.shape[1] * scale), h))
            composite = np.hstack([view, topdown])
        else:
            composite = view

        out_path = os.path.join(args.out, f"step_{step:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        n_explored = sum(1 for _, _, u in a_checked if not u)
        print(f"step {step:3d}: {len(a_initial)} raw rays -> "
              f"{n_explored} explored-trimmed -> {len(final_actions)} proposed "
              f"({out_path})")

        # --- scripted rollout: greedy longest proposed ray, turn around if stuck
        if final_actions:
            r, theta = max(final_actions, key=lambda a: a[0])
            obs = env.step(PolarAction(r, theta))
        else:
            obs = env.step(PolarAction(0.0, math.pi))
        if obs is None:
            print("environment ended the episode")
            break


if __name__ == "__main__":
    main()

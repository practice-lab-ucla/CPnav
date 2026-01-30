from CPnav.env import ObjectNavEnv
import cv2
import matplotlib.pyplot as plt
from CPnav.action_proposer.action_proposer import ActionProposer
from CPnav.projection_on_image import project_actions_on_image  
from CPnav.action_proposer.voxel_map import VoxelMap

import yaml
import numpy as np

with open("config/CPNav.yaml", "r") as f:
    cfg = yaml.safe_load(f)

env = ObjectNavEnv(cfg)

env._initialize_experiment()
env._initialize_episode(0)

obs = env.reset()

rgba = obs["color_sensor"]          # shape (H, W, 4), uint8
rgb = rgba[:, :, :3].copy()         # drop alpha channel and copy

ap = ActionProposer(cfg)
voxel = VoxelMap(cfg)

print("voxelsadsadas",voxel)

a_initial_raw = ap._build_initial_actions(obs)          

a_after_explored = ap._refine_explored(a_initial_raw, obs["agent_state"], voxel)


a_final_full  = ap._refine_actions(a_after_explored)       
actions = [(r, theta) for (r, theta, _) in a_final_full]



# actions.append([1.7, np.deg2rad(34.5)])



# === NEW: project actions onto the image ===
agent_state = obs["agent_state"]
sensor_state = agent_state.sensor_states["color_sensor"]

projected_mapping = project_actions_on_image(
    cfg,
    rgb,
    actions,                # refined actions (same as before)
    agent_state,
    sensor_state,
    chosen_index=None,
    initial_actions=a_initial_raw,  
    show_initial=True
)

print("Projected mapping (action -> label):", projected_mapping)

# Visualize
plt.imshow(rgb)
plt.axis("off")
plt.show()


# === update voxel map and visualize it ===
# Use a_initial_raw for voxel map update (bounded by max_action_dist)
voxel.update_from_actions(
    a_initial_raw,
    agent_state,
    max_action_dist=ap.max_action_dist,
    clip_frac=ap.clip_frac,
)

# Use the *projected* actions to draw arrows/numbers on top-down map
voxel_image = voxel.generate_voxel_image(
    projected_mapping,
    agent_state,
    chosen_action=None,   # or e.g. 1 if you want to highlight action "1"
)

plt.figure()
plt.title("Voxel map (explored = grey, unexplored = green)")
# OpenCV images are BGR, matplotlib expects RGB
plt.imshow(cv2.cvtColor(voxel_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
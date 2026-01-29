from CPnav.env import ObjectNavEnv
import matplotlib.pyplot as plt
from CPnav.action_proposer.action_proposer import ActionProposer
from CPnav.projection_on_image import project_actions_on_image  # <-- NEW

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

actions = ap.propose(obs)  # voxel_map=None for now
# actions.append([1.7, np.deg2rad(34.5)])
print("Actions:", actions)

# === NEW: project actions onto the image ===
agent_state = obs["agent_state"]
sensor_state = agent_state.sensor_states["color_sensor"]

projected_mapping = project_actions_on_image(
    cfg,
    rgb,          # this will be modified in-place
    actions,
    agent_state,
    sensor_state,
    chosen_index=None,  # or e.g. 0 if you want to highlight the first action
)

print("Projected mapping (action -> label):", projected_mapping)

# Visualize
plt.imshow(rgb)
plt.axis("off")
plt.show()

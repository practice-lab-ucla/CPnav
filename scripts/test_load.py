from CPnav.env import ObjectNavEnv
import yaml

with open("config/CPNav.yaml", "r") as f:
    cfg = yaml.safe_load(f)

env = ObjectNavEnv(cfg)

env._initialize_experiment()
env._initialize_episode(0)
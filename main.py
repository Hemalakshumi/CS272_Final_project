import time
import numpy as np
from your_env2 import YourEnv

env = YourEnv(render_mode="human")
obs = env.reset()

for _ in range(500):  # run 500 steps
    # Sample random action
    action = env.action_space.sample()
    
    # Convert numpy array to dictionary for YourEnv
    if isinstance(action, np.ndarray):
        action = {"steering": float(action[0]), "acceleration": float(action[1])}

    # Use the public step() method, not _step()
    obs, reward, terminated, truncated, info = env._step(action)
    
    env.render()
    time.sleep(0.05)

    if terminated or truncated:
        obs = env.reset()

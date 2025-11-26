# test_merge.py  (Google Colab friendly)

import gymnasium as gym
import numpy as np
import highway_env
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
from stable_baselines3 import DQN
from custom_cnn import CustomCNN

# Same config as training
config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],
        "scaling": 1.75,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 2,
    "vehicles_count": 20,
    "policy_frequency": 2,
}

# Colab-friendly rendering: rgb_array
env = gym.make("merge-v0", config=config, render_mode="rgb_array")

# Load trained DQN model
model = DQN.load("dqn_merge_grayscale", env=env)

rewards = []

for ep in range(500):   
    obs, info = env.reset()
    done = False
    ep_reward = 0
    crashes = 0
    lane_changes = 0

    step_count = 0
    while not done:
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        # Count crashes and lane changes
        crashes += int(info.get("crashed", 0))
        lane_changes += int(info.get("is_lane_change", 0))

        # Render every 5 steps to speed up Colab
        if step_count % 5 == 0:
            frame = env.render()
            plt.imshow(frame)
            plt.axis("off")
            clear_output(wait=True)
            plt.show()
            time.sleep(0.10)  # ~20 FPS, adjust as needed

        step_count += 1
        done = terminated or truncated
        ep_reward += reward

    print(f"Episode {ep}: reward = {ep_reward}, crashes = {crashes}, lane_changes = {lane_changes}")
    rewards.append(ep_reward)

env.close()

print("Mean reward across episodes:", np.mean(rewards))

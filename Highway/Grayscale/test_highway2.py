import gymnasium as gym
import numpy as np
import highway_env
import cv2
import os

from gymnasium.wrappers import RecordVideo
from custom_ppo import CustomPPO
from custom_cnn import CustomCNN


# --------------------------
# Environment Configuration
# --------------------------
config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],
        "scaling": 1.75,
    },
    "policy_frequency": 2,
    "action": {"type": "DiscreteMetaAction"},
}


# --------------------------
# VIDEO RECORDING SETTINGS
# --------------------------
def record_trigger(episode_num):
    # Record only every 50 episodes
    return episode_num % 50 == 0
  


video_folder = "videos/"
os.makedirs(video_folder, exist_ok=True)


# --------------------------
# Make Environment
# IMPORTANT: render_mode must be "rgb_array" for RecordVideo
# --------------------------
env = gym.make("highway-v0", config=config, render_mode="rgb_array")
env = RecordVideo(env, video_folder, episode_trigger=record_trigger)


# --------------------------
# Load Model
# --------------------------
model = CustomPPO.load("ppo_highway_grayscale_meta", env=env)


# --------------------------
# Evaluation Loop
# --------------------------
rewards = []

for ep in range(500):
    obs, info = env.reset()
    done = False
    ep_reward = 0

    print(f"Episode {ep} started... (Recording: {record_trigger(ep)})")

    while not done:
        # Model prediction
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        # ----------- LIVE RENDER -----------
        frame = env.render()
        if frame is not None:
            cv2.imshow("Highway PPO Evaluation", frame)
            cv2.waitKey(1)

    print(f"Episode {ep} reward: {ep_reward}")
    rewards.append(ep_reward)


# --------------------------
# Cleanup
# --------------------------
env.close()
cv2.destroyAllWindows()

print("\nEvaluation complete.")
print("Mean reward over 500 episodes:", np.mean(rewards))

# Save rewards if needed
np.save("test_rewards_500.npy", np.array(rewards))

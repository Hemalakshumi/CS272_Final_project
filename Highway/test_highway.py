import gymnasium as gym
import numpy as np
import highway_env
from custom_ppo import CustomPPO
from custom_cnn import CustomCNN

config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
    "policy_frequency": 2
    , "action": {"type": "DiscreteMetaAction"},
}

env = gym.make("highway-v0", config=config, render_mode="human")

model = CustomPPO.load("ppo_highway_grayscale_meta", env=env)

rewards = []

for ep in range(500):
    obs, info = env.reset()
    done = False
    ep_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)  
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated
        ep_reward += reward

    print(ep, ep_reward)
    rewards.append(ep_reward)

env.close()
print("Mean:", np.mean(rewards))

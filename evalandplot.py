# evaluate_highway_agent.py
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import sys

sys.modules["gym"] = gym  # gymnasium compatibility fix

def make_env():
    env = gym.make("highway-v0", render_mode="human")
    env.configure({
        "observation": {
            "type": "GrayscaleObservation",
            "weights": [0.2989, 0.5870, 0.1140],
            "stack_size": 4,
            "observation_shape": (84, 84)
        },
        "screen_width": 600,
        "screen_height": 150,
    })
    env.reset()
    return env

env = DummyVecEnv([make_env])
env = VecTransposeImage(env)

model = PPO.load("runs/highway_grayscale/ppo_highway_grayscale", env=env)
obs = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done.any():
        obs = env.reset()

env.close()

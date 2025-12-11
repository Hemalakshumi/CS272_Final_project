import os
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure CustomMLP is importable
import optimized_highway_lidar  # noqa: F401


BASE_DIR = "model_checkpoints_highway_optimized"
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


HIGHWAY_CONFIG = {
    "observation": {
        "type": "LidarObservation",
        "cells": 128,
        "range": 64,
        "normalise": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,
    "initial_spacing": 2,
    "collision_reward": -1,
    "reward_speed_range": [20, 30],
    "simulation_frequency": 15,
    "policy_frequency": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
}


def make_env():
    env = gym.make("highway-v0")
    env.unwrapped.configure(HIGHWAY_CONFIG)
    env.reset()
    return env


def main():
    # Raw DummyVecEnv (no VecNormalize)
    venv = DummyVecEnv([make_env])

    model_path = os.path.join(MODEL_DIR, "ppo_highway_lidar_optimized_step_100000.zip")
    model = PPO.load(model_path, env=venv)
    print(f"Loaded model: {model_path}")

    n_episodes = 50
    rewards = []

    for ep in range(n_episodes):
        obs = venv.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, info = venv.step(action)

            total_reward += float(reward[0])
            steps += 1
            done = bool(done_vec[0])

            if steps > 2000:
                print(f"Episode {ep}: forced stop at 2000 steps")
                break

        print(f"Episode {ep} reward (raw): {total_reward}")
        rewards.append(total_reward)

    rewards = np.array(rewards, dtype=np.float32)

    out_path = os.path.join(LOG_DIR, "highway_test_rewards_optimized.txt")
    np.savetxt(out_path, rewards)

    print(f"\nSaved test rewards → {out_path}")
    print(
        f"Test results over {n_episodes} episodes: "
        f"mean = {rewards.mean():.2f}, std = {rewards.std():.2f}"
    )


if __name__ == "__main__":
    main()

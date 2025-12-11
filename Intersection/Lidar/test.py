import os
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure CustomMLP is importable (from training script)
import optimized_intersection_lidar  # noqa: F401

from highway_env.envs.intersection_env import IntersectionEnv


# ---------------------------------------------------
# Config (same as training, without RewardWrapper)
# ---------------------------------------------------
INTERSECTION_CONFIG = {
    "observation": {
        "type": "LidarObservation",
        "cells": 128,
        "range": 64,
        "normalise": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": False,
        "lateral": True,
    },
    "duration": 13,
    "destination": "o1",
    "initial_vehicle_count": 10,
    "spawn_probability": 0.6,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.6],
    "scaling": 5.5 * 1.3,
    "collision_reward": -1,
    "normalize_reward": False,
}


def make_env():
    env = gym.make("intersection-v0")
    env.unwrapped.configure(INTERSECTION_CONFIG)
    env.reset()
    return env


def main():
    base_dir = "model_checkpoints_lidar_optimized"
    model_dir = os.path.join(base_dir, "models")
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    venv = DummyVecEnv([make_env])

    model_path = os.path.join(model_dir, "ppo_intersection_lidar_optimized_step_175000.zip")
    model = PPO.load(model_path, env=venv)
    print(f"Loaded model: {model_path}")

    n_episodes = 500
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
    out_path = os.path.join(log_dir, "intersection_test_rewards_optimized.txt")
    np.savetxt(out_path, rewards)

    print(f"\nSaved test rewards to: {out_path}")
    print(
        f"Test results over {n_episodes} episodes — "
        f"mean: {rewards.mean():.2f}, std: {rewards.std():.2f}"
    )


if __name__ == "__main__":
    main()

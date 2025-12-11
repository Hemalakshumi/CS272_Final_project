import os
import numpy as np
import gymnasium as gym
import highway_env 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Base project directory (for Colab / Google Drive use)
# ----------------------------------------------------
BASE_DIR = "/content/drive/MyDrive/RL_project_1"
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


# Environment configuration 
# ----------------------------------------------------
BASE_CONFIG = {
    "observation": {
        "type": "LidarObservation",
        "cells": 128,
        "range": 64,
        "normalise": True,
    },
    "action": {"type": "DiscreteMetaAction"},
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
    """
    Create a raw merge-v0 environment using BASE_CONFIG.
    Reward shaping is intentionally omitted for evaluation.
    """
    env = gym.make("merge-v0")
    env.unwrapped.configure(BASE_CONFIG)
    env.reset()
    return env


def main():
    # Step 1: Create an unwrapped environment for evaluation
    venv = DummyVecEnv([make_env])

    # Step 2: Load stored VecNormalize statistics and wrap the environment
    vecnorm_path = os.path.join(MODEL_DIR, "merge_vecnormalize.pkl")
    venv = VecNormalize.load(vecnorm_path, venv)
    print(f"Loaded VecNormalize statistics from: {vecnorm_path}")

    # Disable reward normalization for raw reward evaluation
    venv.training = False
    venv.norm_reward = False

    # Step 3: Load trained PPO model
    model_path = os.path.join(MODEL_DIR, "ppo_merge_optimized")
    model = PPO.load(model_path, env=venv)
    print(f"Loaded optimized model from: {model_path}")

    # Step 4: Evaluate for multiple episodes using raw rewards
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

            done = bool(done_vec[0])
            total_reward += float(reward[0])
            steps += 1

            # Safety cap to avoid infinite episodes
            if steps > 2000:
                print(f"Episode {ep}: forced stop at 2000 steps")
                break

        print(f"Episode {ep} reward (raw): {total_reward}")
        rewards.append(total_reward)

    rewards = np.array(rewards, dtype=np.float32)

    # Step 5: Save evaluation results
    out_path = os.path.join(LOG_DIR, "test_rewards_optimized.txt")
    np.savetxt(out_path, rewards)

    print(f"\nSaved evaluation rewards to: {out_path}")
    print(
        f"Evaluation results over {n_episodes} episodes: "
        f"mean = {rewards.mean():.2f}, std = {rewards.std():.2f}"
    )


if __name__ == "__main__":
    main()

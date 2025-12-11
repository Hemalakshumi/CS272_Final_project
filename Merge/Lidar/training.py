import os
import numpy as np
import gymnasium as gym
import highway_env  
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



# Shared environment configuration for merge-v0

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
    """Create a merge-v0 environment using BASE_CONFIG."""
    env = gym.make("merge-v0")
    env.unwrapped.configure(BASE_CONFIG)
    env.reset()
    return env



# Custom feature extractor 
# -----------------------------------------------------
class CustomMLP(BaseFeaturesExtractor):
    """
    Feature extractor for LidarObservation of shape (72 x 2).
    Performs: flatten → 256 → features_dim.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_inputs = int(np.prod(observation_space.shape))

        self.model = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        obs = th.flatten(obs, start_dim=1)
        return self.model(obs)



# Reward shaping: adds a small survival bonus
# ------------------------------------------------------
class RewardWrapper(gym.RewardWrapper):
    """
    Adds a constant bonus to each reward during training.
    Evaluation is always performed on raw environment rewards.
    """

    def reward(self, reward):
        return reward + 0.05


# Callback to record episodic (normalized) returns
# ------------------------------------------------------
class EpisodicReturnCallback(BaseCallback):
    """
    Collects episodic returns based on normalized rewards output by VecNormalize.

    One return value is stored per completed episode.
    """

    def __init__(self, log_path: str, save_every_episodes: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.save_every_episodes = save_every_episodes

        self.current_returns = None   # Accumulates per-env episode returns
        self.episode_returns = []     # List of float episode returns
        self.episode_count = 0

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        if self.current_returns is None:
            self.current_returns = np.zeros_like(rewards, dtype=np.float32)

        self.current_returns += rewards

        for i, done in enumerate(dones):
            if done:
                ep_ret = float(self.current_returns[i])
                self.episode_returns.append(ep_ret)
                self.current_returns[i] = 0.0
                self.episode_count += 1

                if self.episode_count % self.save_every_episodes == 0:
                    np.save(self.log_path, np.array(self.episode_returns, dtype=np.float32))
                    if self.verbose > 0:
                        print(f"[Callback] Saved {self.episode_count} episodes to {self.log_path}")

        return True

    def _on_training_end(self) -> None:
        np.save(self.log_path, np.array(self.episode_returns, dtype=np.float32))
        print(
            f"Saved normalized training episodic returns to {self.log_path} "
            f"(episodes: {len(self.episode_returns)})"
        )



# Evaluation using raw environment reward scale
# ------------------------------------------------------
def evaluate_raw_with_vecenv(model, vec_env: VecNormalize, n_episodes: int = 20):
    """
    Evaluate the policy using raw (unnormalized) environment rewards.

    VecNormalize is still used for observations, but reward normalization is disabled.
    """
    vec_env.training = False
    vec_env.norm_reward = False

    episode_returns = []
    n_envs = vec_env.num_envs

    obs = vec_env.reset()
    ep_return = np.zeros(n_envs, dtype=np.float32)

    while len(episode_returns) < n_episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = vec_env.step(actions)
        ep_return += rewards

        for i in range(n_envs):
            if dones[i]:
                episode_returns.append(ep_return[i])
                ep_return[i] = 0.0
                if len(episode_returns) >= n_episodes:
                    break

    episode_returns = np.array(episode_returns, dtype=np.float32)
    print(
        f"[Evaluation] Raw-reward evaluation over {n_episodes} episodes:"
        f" mean = {episode_returns.mean():.2f}, std = {episode_returns.std():.2f}"
    )


def main():
    base_dir = "/content/drive/MyDrive/RL_project_1"
    models_dir = os.path.join(base_dir, "models")
    logs_dir = os.path.join(base_dir, "logs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, "training_rewards_normalized.npy")

    def make_wrapped_env():
        env = make_env()
        env = RewardWrapper(env)
        return env

    train_env = DummyVecEnv([make_wrapped_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=5.0)

    policy_kwargs = dict(
        features_extractor_class=CustomMLP,
        features_extractor_kwargs={"features_dim": 256},
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1.5e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.995,
        gae_lambda=0.95,
        n_epochs=10,
        ent_coef=0.005,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cpu",
    )

    callback = EpisodicReturnCallback(log_path=log_path, save_every_episodes=10, verbose=1)

    print("Training PPO (250k timesteps)...")
    model.learn(total_timesteps=250_000, callback=callback)
    print("Training completed.")

    model_path = os.path.join(models_dir, "ppo_merge_optimized")
    norm_path = os.path.join(models_dir, "merge_vecnormalize.pkl")

    model.save(model_path)
    train_env.save(norm_path)

    print(f"Saved optimized model to {model_path}.zip")
    print(f"Saved VecNormalize statistics to {norm_path}")

    evaluate_raw_with_vecenv(model, train_env, n_episodes=20)


if __name__ == "__main__":
    main()

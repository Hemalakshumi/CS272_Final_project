import os
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from highway_env.envs.intersection_env import IntersectionEnv

# Get default collision reward, fallback -1.0 if missing
DEFAULT_COLLISION_REWARD = getattr(IntersectionEnv, "COLLISION_REWARD", -1.0)


# -------------------------
# Intersection-specific config
# -------------------------
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
    "duration": 13,  # [s]
    "destination": "o1",
    "initial_vehicle_count": 10,
    "spawn_probability": 0.6,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.6],
    "scaling": 5.5 * 1.3,
    "collision_reward": DEFAULT_COLLISION_REWARD,
    "normalize_reward": False,
}


def make_env():
    """Create intersection-v0 env with INTERSECTION_CONFIG."""
    env = gym.make("intersection-v0")
    env.unwrapped.configure(INTERSECTION_CONFIG)
    env.reset()
    return env


# -------------------------
# Custom feature extractor
# -------------------------
class CustomMLP(BaseFeaturesExtractor):
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


# -------------------------
# Reward shaping wrapper
# -------------------------
class RewardWrapper(gym.RewardWrapper):
    """Add a small survival bonus to the reward."""

    def reward(self, reward):
        return reward + 0.05


# -------------------------
# Episodic return logger
# -------------------------
class EpisodicReturnCallback(BaseCallback):
    """Log episodic returns (raw reward)."""

    def __init__(self, log_path: str, save_every_episodes: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.save_every_episodes = save_every_episodes
        self.current_returns = None
        self.episode_returns = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]  # raw rewards
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
                        print(
                            f"[Intersection-Callback] Saved {self.episode_count} episodes "
                            f"to {self.log_path}"
                        )
        return True

    def _on_training_end(self) -> None:
        np.save(self.log_path, np.array(self.episode_returns, dtype=np.float32))
        print(
            f"[Intersection] Saved training episodic returns to {self.log_path} "
            f"(episodes: {len(self.episode_returns)})"
        )


# -------------------------
# Checkpoint callback
# -------------------------
class ProgressCheckpointCallback(BaseCallback):
    """
    Save:
      - Checkpoints at 10%, 20%, ..., 100% of training
      - A 'latest' checkpoint that is overwritten each time
    """

    def __init__(
        self,
        total_timesteps: int,
        models_dir: str,
        prefix: str = "ppo_intersection_lidar_optimized",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.models_dir = models_dir
        self.prefix = prefix

        os.makedirs(self.models_dir, exist_ok=True)

        fractions = [0.1 * i for i in range(1, 11)]  # 0.1, 0.2, ..., 1.0
        self.checkpoints = sorted({int(frac * self.total_timesteps) for frac in fractions})
        self.saved = set()

    def _on_step(self) -> bool:
        current = self.num_timesteps

        for ckpt in self.checkpoints:
            if current >= ckpt and ckpt not in self.saved:
                ckpt_path = os.path.join(self.models_dir, f"{self.prefix}_step_{ckpt}")
                latest_path = os.path.join(self.models_dir, f"{self.prefix}_latest")

                self.model.save(ckpt_path)
                self.model.save(latest_path)

                self.saved.add(ckpt)
                if self.verbose > 0:
                    print(
                        f"[Checkpoint] Saved checkpoint at step {ckpt} / "
                        f"{self.total_timesteps} to {ckpt_path}"
                    )
        return True


# -------------------------
# Evaluation (raw reward)
# -------------------------
def evaluate_raw_with_vecenv(model, vec_env: VecNormalize, n_episodes: int = 20):
    """Evaluate the model with raw (unnormalized) rewards."""
    vec_env.training = False
    vec_env.norm_reward = False

    episode_returns = []
    n_envs = vec_env.num_envs
    obs = vec_env.reset()
    ep_return = np.zeros(n_envs, dtype=np.float32)

    while len(episode_returns) < n_episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(actions)
        ep_return += rewards

        for i in range(n_envs):
            if dones[i]:
                episode_returns.append(ep_return[i])
                ep_return[i] = 0.0
                if len(episode_returns) >= n_episodes:
                    break

    episode_returns = np.array(episode_returns, dtype=np.float32)
    print(
        f"[Intersection-Eval] Raw eval over {n_episodes} episodes: "
        f"mean_reward = {episode_returns.mean():.2f}, std = {episode_returns.std():.2f}"
    )


def main():
    BASE_DIR = "model_checkpoints_lidar_optimized"
    models_dir = os.path.join(BASE_DIR, "models")
    logs_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    total_timesteps = 250_000

    # Episodic rewards log
    log_path = os.path.join(logs_dir, "intersection_training_rewards.npy")

    def make_wrapped_env():
        env = make_env()
        env = RewardWrapper(env)
        return env

    # Single-env DummyVecEnv
    train_env = DummyVecEnv([make_wrapped_env])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,      # normalize observations
        norm_reward=False,  # keep rewards raw
        clip_obs=5.0,
    )

    policy_kwargs = dict(
        features_extractor_class=CustomMLP,
        features_extractor_kwargs=dict(features_dim=256),
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

    episodic_cb = EpisodicReturnCallback(
        log_path=log_path,
        save_every_episodes=10,
        verbose=1,
    )

    ckpt_cb = ProgressCheckpointCallback(
        total_timesteps=total_timesteps,
        models_dir=models_dir,
        prefix="ppo_intersection_lidar_optimized",
        verbose=1,
    )

    callback = CallbackList([episodic_cb, ckpt_cb])

    print("[Intersection] Training PPO (250k steps, raw rewards, CPU)...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    print("[Intersection] Training complete.")

    model_path = os.path.join(models_dir, "ppo_intersection_lidar_optimized")
    norm_path = os.path.join(models_dir, "intersection_lidar_vecnormalize.pkl")

    model.save(model_path)
    train_env.save(norm_path)
    print(f"[Intersection] Saved final model to {model_path}.zip")
    print(f"[Intersection] Saved VecNormalize stats (obs only) to {norm_path}")

    evaluate_raw_with_vecenv(model, train_env, n_episodes=20)


if __name__ == "__main__":
    main()

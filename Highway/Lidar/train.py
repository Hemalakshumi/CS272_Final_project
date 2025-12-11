import os
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ==============================
# Highway LIDAR configuration
# ==============================
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


# ==============================
# LiDAR feature extractor
# ==============================
class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_inputs = int(np.prod(observation_space.shape))

        self.model = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.model(th.flatten(obs, start_dim=1))


# ==============================
# Reward wrapper
# ==============================
class RewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return reward + 0.05


# ==============================
# Episodic return logger
# ==============================
class EpisodicReturnCallback(BaseCallback):
    def __init__(self, log_path, save_every_episodes=25, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.save_every = save_every_episodes
        self.current = None
        self.returns = []
        self.count = 0

    def _on_step(self):
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        if self.current is None:
            self.current = np.zeros_like(rewards, dtype=np.float32)

        self.current += rewards

        for i, done in enumerate(dones):
            if done:
                self.returns.append(float(self.current[i]))
                self.current[i] = 0.0
                self.count += 1

                if self.count % self.save_every == 0:
                    np.save(self.log_path, np.array(self.returns, dtype=np.float32))
                    if self.verbose > 0:
                        print(f"[Highway] Saved {self.count} episodes → {self.log_path}")

        return True

    def _on_training_end(self):
        np.save(self.log_path, np.array(self.returns, dtype=np.float32))
        print(f"[Highway] Saved episodic returns → {self.log_path} ({len(self.returns)})")


# ==============================
# Progress checkpoint callback
# ==============================
class ProgressCheckpointCallback(BaseCallback):
    """
    Saves checkpoints at 10%, 20%, ..., 100% of total timesteps
    and maintains a continuously updated 'latest' checkpoint.
    """

    def __init__(self, total_timesteps, models_dir, prefix, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.models_dir = models_dir
        self.prefix = prefix
        os.makedirs(self.models_dir, exist_ok=True)

        fractions = [0.1 * i for i in range(1, 11)]
        self.checkpoints = sorted({int(frac * self.total_timesteps) for frac in fractions})
        self.saved = set()

    def _on_step(self):
        current = self.num_timesteps

        for ckpt in self.checkpoints:
            if current >= ckpt and ckpt not in self.saved:
                step_path = os.path.join(self.models_dir, f"{self.prefix}_step_{ckpt}")
                latest_path = os.path.join(self.models_dir, f"{self.prefix}_latest")

                self.model.save(step_path)
                self.model.save(latest_path)
                self.saved.add(ckpt)

                if self.verbose > 0:
                    print(f"[Highway] Saved checkpoint at {ckpt} steps → {step_path}")

        return True


# ==============================
# Evaluation
# ==============================
def evaluate_raw_with_vecenv(model, vec_env, n_episodes=20):
    vec_env.training = False
    vec_env.norm_reward = False

    returns = []
    obs = vec_env.reset()
    ep_ret = np.zeros(vec_env.num_envs, dtype=np.float32)

    while len(returns) < n_episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = vec_env.step(actions)
        ep_ret += rewards

        for i, done in enumerate(dones):
            if done:
                returns.append(ep_ret[i])
                ep_ret[i] = 0.0
                if len(returns) >= n_episodes:
                    break

    returns = np.array(returns, dtype=np.float32)
    print(f"[Highway Evaluation] mean={returns.mean():.2f}, std={returns.std():.2f}")


# ==============================
# Main training procedure
# ==============================
def main():
    base_dir = "model_checkpoints_highway_optimized"
    models_dir = os.path.join(base_dir, "models")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, "highway_training_rewards.npy")

    def make_wrapped():
        env = make_env()
        env = RewardWrapper(env)
        return env

    # 4 parallel envs
    train_env = SubprocVecEnv([make_wrapped for _ in range(4)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=5.0,
        clip_reward=10.0,
    )

    policy_kwargs = dict(
        features_extractor_class=CustomMLP,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    total_timesteps = 250_000

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device="auto",
        verbose=1,
    )

    episodic_cb = EpisodicReturnCallback(log_path, save_every_episodes=25, verbose=1)
    ckpt_cb = ProgressCheckpointCallback(
        total_timesteps,
        models_dir,
        prefix="ppo_highway_lidar_optimized",
        verbose=1,
    )

    callback = CallbackList([episodic_cb, ckpt_cb])

    print("[Highway] Training PPO (250k steps, normalized rewards)...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    print("[Highway] Training complete.")

    model_path = os.path.join(models_dir, "ppo_highway_lidar_optimized")
    norm_path = os.path.join(models_dir, "highway_lidar_vecnormalize.pkl")

    model.save(model_path)
    train_env.save(norm_path)

    print(f"[Highway] Saved model → {model_path}")
    print(f"[Highway] Saved VecNormalize → {norm_path}")

    evaluate_raw_with_vecenv(model, train_env, n_episodes=20)


if __name__ == "__main__":
    main()

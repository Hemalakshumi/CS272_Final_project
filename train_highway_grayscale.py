"""
Train a PPO agent on the highway-v0 environment using feature (vector) observations.
Compatible with:
    - gymnasium >= 0.28
    - highway-env (latest)
    - stable-baselines3 >= 2.2
"""

import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os
import sys

# Compatibility alias: highway-env still imports 'gym' internally
sys.modules["gym"] = gym


def make_env():
    """Create and configure the Highway environment."""
    env = gym.make("highway-v0", render_mode=None)
    env.configure({
        "observation": {
            "type": "Kinematics",   # default feature-based observations (shape (5,5))
        },
        "policy_frequency": 15,     # frequency of decisions
        "screen_width": 600,
        "screen_height": 150,
        "duration": 40,             # episode length (in seconds)
        "vehicles_count": 20,       # number of cars on the road
        "offroad_terminal": True,   # terminate when agent goes off road
    })
    env.reset()
    return env


if __name__ == "__main__":
    logdir = "runs/highway_mlp"
    os.makedirs(logdir, exist_ok=True)

    # Create vectorized environment for SB3
    env = DummyVecEnv([make_env])

    # PPO model with MLP policy (for non-image observations)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=logdir,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        n_epochs=10,
        device="auto",
    )

    # Evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(logdir, "best_model"),
        log_path=os.path.join(logdir, "eval"),
        eval_freq=10_000,           # Evaluate every 10k steps
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    total_timesteps = 200_000  # Adjust as needed
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save(os.path.join(logdir, "ppo_highway_mlp"))
    print("✅ Training finished! Model saved to:", logdir)

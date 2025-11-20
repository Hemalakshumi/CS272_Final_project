import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

import your_env   # This registers YourEnv

def train():
    logdir = "logs_parallel/"
    os.makedirs(logdir, exist_ok=True)

    # Create single env (NO make_vec_env to avoid observation errors)
    env = gym.make("your_env/YourEnv-v0") if hasattr(gym.envs.registry, "env_specs") \
        else your_env.YourEnv()

    env = Monitor(env)

    # TensorBoard + stdout logger
    logger = configure(logdir, ["stdout", "tensorboard"])

    # Lightweight PPO config for fast training (15 min)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=logdir,
        device="cpu"
    )
    model.set_logger(logger)

    # Only 50k steps → manageable & quick
    model.learn(total_timesteps=50_000)

    model.save("ppo_parking_short")
    print("Training completed. Saved model ppo_parking_short.zip")

if __name__ == "__main__":
    train()

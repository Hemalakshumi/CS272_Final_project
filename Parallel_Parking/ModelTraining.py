# -------------------------------
# 1️⃣ Imports
# -------------------------------
import os
import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from gymnasium.wrappers import TransformReward
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit

# this has been run on google Colab
def make_env():
    env = ParallelParkingEnv(render_mode="rgb_array")
    env = SteeringRewardWrapper(env)
    env = Monitor(env)
    return env

policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256],
        qf=[256, 256]
    )
)

vec_env = make_vec_env(
    lambda: TimeLimit(
        SteeringRewardWrapper(ParallelParkingEnv(render_mode="rgb_array")),
        max_episode_steps=200
    ),
    n_envs=16
)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

eval_env = make_vec_env(
    lambda: TimeLimit(
        SteeringRewardWrapper(ParallelParkingEnv(render_mode="rgb_array")),
        max_episode_steps=200
    ),
    n_envs=8
)
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

model = SAC(
    "MultiInputPolicy",
    env=vec_env,
    ent_coef="auto",
    target_entropy = -3.0,
    n_steps=2048
learning_rate=3e-4,
tau=0.005,
batch_size=256,
gamma=0.995,
buffer_size=200000,
policy_kwargs=policy_kwargs,
verbose=1
)

os.makedirs("./logs/best_model/", exist_ok=True)
os.makedirs("./logs/test_results/", exist_ok=True)
os.makedirs("/content/drive/MyDrive/ParallelParkingModels", exist_ok=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/test_results/",
    eval_freq=10000,
    deterministic=True,
    render=False
)
save_path = "/content/drive/MyDrive/ParallelParkingModels"
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=save_path,
    name_prefix="sac_parallelparking"
)
callback = CallbackList([checkpoint_callback,eval_callback])

T = 350_000
C = 100_000

for i in range(T // C):
    print(f"Training chunk {i+1}/{T // C}")
    model.learn(
        total_timesteps=C,
        reset_num_timesteps=False,
        callback=callback
    )
    model.save(os.path.join(save_path, f"sac_parallelparking_chunk{i+1}"))

print(" Training completed. Models are saved.")

vecnormalize_path = os.path.join(save_path, "vecnormalize.pkl")
vec_env.save(vecnormalize_path)
print(f" VecNormalize saved at: {vecnormalize_path}")

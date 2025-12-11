
import os
import glob
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

def make_env_train(seed=None, monitor_path=None):
    def _init():
        env = EmergencyEnv()

        if monitor_path:
            env = Monitor(env, filename=monitor_path)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return _init

def make_env_eval(seed=None, monitor_path=None):
    def _init():
        env = EmergencyEnv_eval()
        if monitor_path:
            env = Monitor(env, filename=monitor_path)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return _init


save_path = "/content/drive/MyDrive/EmergencyVehicles"
os.makedirs(save_path, exist_ok=True)
os.makedirs("./logs/best_model/", exist_ok=True)
os.makedirs("./logs/training/", exist_ok=True)
os.makedirs("./logs/performance_test/", exist_ok=True)


num_envs = 8
vec_env = SubprocVecEnv([make_env_train(i, monitor_path=f"./logs/training/monitor_{i}.csv")
                         for i in range(num_envs)])
#vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
vec_env = VecNormalize.load("/content/drive/MyDrive/EmergencyVehicles/vecnormalize.pkl", vec_env)



policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=3e-4,
    batch_size=1024,
    gamma=0.995,
    n_steps=4096,
    device="cuda",
    verbose=1,
    gae_lambda=0.95,
    ent_coef=0.01,
    policy_kwargs=policy_kwargs
)


checkpoint_path = "/content/drive/MyDrive/EmergencyVehicles/ppo_emergency_chunk7.zip"
model = PPO.load(checkpoint_path, verbose=1)
PPO.learning_rate=5e-5
model.set_env(vec_env)


checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path=save_path,
    name_prefix="ppo_emergencyvehicles"
)


eval_monitor_path = "./logs/performance_test/eval_monitor.csv"
eval_env = DummyVecEnv([make_env_eval(999, monitor_path=eval_monitor_path)])


if os.path.exists(os.path.join(save_path, "vecnormalize.pkl")):
    eval_env = VecNormalize.load(os.path.join(save_path, "vecnormalize.pkl"), eval_env)
else:
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

eval_env.training = False
eval_env.norm_reward = False

# EvalCallback to evaluate during training
eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/performance_test/",
    eval_freq=1000,  # evaluate every 25 steps (or timesteps)
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

callback = CallbackList([checkpoint_callback, eval_callback])


TOTAL_STEPS = 50_000
CHUNK_STEPS = 25_000

for i in range(TOTAL_STEPS // CHUNK_STEPS):
    print(f"Training chunk {i+1+8}/{TOTAL_STEPS // CHUNK_STEPS}")
    model.learn(
        total_timesteps=CHUNK_STEPS,
        reset_num_timesteps=False,
        callback=callback
    )
    model.save(os.path.join(save_path, f"ppo_emergency_chunk{i+1+8}"))


vec_env.save(os.path.join(save_path, "vecnormalize.pkl"))
print(" Training completed and VecNormalize saved")



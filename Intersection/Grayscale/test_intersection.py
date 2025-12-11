# test_intersection.py
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from make_intersection_env import make_intersection_env_fn

# load trained model
model = PPO.load("./ppo_intersection_logs/ppo_intersection_final") 

# create a simple test env (non-vectorized)
env = make_intersection_env_fn({
"observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],
        "scaling": 1.75,
    },    "action": {"type": "DiscreteMetaAction"},
    "duration": 120,
}, render_mode="rgb_array")()
env = Monitor(env, filename="./intersection_test_monitor.csv")

# run N episodes and record rewards
n_eval = 500
rewards = []
for ep in range(n_eval):
    obs, info = env.reset()
    done = False
    ep_r = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        ep_r += float(r)
    rewards.append(ep_r)
    print(f"Episode {ep}: reward={ep_r}")

env.close()
print("Mean reward:", np.mean(rewards))

# -------- plotting (violin) --------
plt.figure(figsize=(6,4))
plt.violinplot(rewards, showmeans=True)
plt.title("Intersection evaluation (violin)")
plt.ylabel("Reward")
plt.show()

# -------- learning curve from training monitor files --------
monitor_files = sorted(glob.glob("./ppo_intersection_logs/monitor/monitor_*.csv.monitor.csv"))
all_data = []
for f in monitor_files:
    df = pd.read_csv(f, skiprows=1)
    all_data.append(df)
if all_data:
    df_all = pd.concat(all_data, ignore_index=True)
    plt.figure(figsize=(10,4))
    plt.plot(df_all["l"].cumsum(), df_all["r"].rolling(10).mean())  # crude
    plt.xlabel("Episodes (cumulative)")
    plt.ylabel("Episode reward (rolling mean)")
    plt.title("Training learning curve (approx)")
    plt.show()
else:
    print("No monitor files found in ./ppo_intersection_logs/")

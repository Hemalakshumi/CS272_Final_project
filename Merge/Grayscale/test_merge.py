import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3 import PPO
from merge_reward_wrapper import MergeRewardWrapper
from make_env import env_config
# ---- Build test env identical to training ----
def make_test_env():
    env = gym.make("merge-v0", config=env_config, render_mode="rgb_array")
    env = MergeRewardWrapper(env)
    return env

dummy = DummyVecEnv([make_test_env])

# ---- Load normalization stats ----
vec_env = VecNormalize.load("merge_vecnorm.pkl", dummy)
vec_env.training = False
vec_env.norm_reward = False

# ---- Load PPO model ----
model = PPO.load("ppo_merge_wrapped", env=vec_env)

# ---- Test ----
rewards = []
for ep in range(50):
    obs = vec_env.reset()
    done = False
    ep_reward = 0
    crashes = 0
    lane_changes = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        info0 = info[0]       # unwrap the info list
        crashes += int(info0.get("crashed", 0))
        lane_changes += int(info0.get("is_lane_change", 0))



        frame = vec_env.venv.envs[0].render()
        plt.imshow(frame)
        plt.axis("off")
        clear_output(wait=True)
        plt.show()
        time.sleep(0.2)
        ep_reward += reward

    print(f"EP {ep}: Reward={ep_reward}, Crashes={crashes}, LaneChanges={lane_changes}")
    rewards.append(ep_reward)

print("Mean reward:", np.mean(rewards))

#make_env.py
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from merge_reward_wrapper import MergeRewardWrapper

def make_merge_env(config, monitor_dir=None, rank=0):
    def _init():
        env = gym.make("merge-v0", config=config)
        env = MergeRewardWrapper(env)
        env = TimeLimit(env, max_episode_steps=200)

        if monitor_dir is not None:
            env = Monitor(env, filename=f"{monitor_dir}/monitor_{rank}.csv")

        return env
    return _init


def create_vec_env(config, monitor_dir="./monitor"):
    env_fns = [
        make_merge_env(config, monitor_dir=monitor_dir, rank=i)
        for i in range(8)
    ]
    vec_env = DummyVecEnv(env_fns)

    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    return vec_env


#merge_reward_wrapper.py
from gymnasium import Wrapper
import numpy as np

class MergeRewardWrapper(Wrapper):
    """
    Reward shaping for MERGE-v0:
      • Encourage correct merge angle
      • Reward larger safe gap
      • Penalize collisions strongly
    """

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Always unwrap highway-env
        env = self.env.unwrapped
        ego = env.vehicle

        # -----------------------
        # Merge angle reward
        # -----------------------
        heading = ego.heading
        target_heading = 0.0   # MERGE highway is mostly horizontal
        angle_diff = abs((heading - target_heading + np.pi) % (2*np.pi) - np.pi)

        angle_reward = np.exp(-(angle_diff / 0.5) ** 2)   # smooth reward peak

        # -----------------------
        # Safe gap reward
        # -----------------------
        min_dist = float("inf")
        for v in env.road.vehicles:
            if v is ego:
                continue
            dist = np.linalg.norm(v.position - ego.position)
            min_dist = min(min_dist, dist)

        gap_reward = np.tanh(min_dist / 12.0)   # normalized 0–1

        # -----------------------
        # Crash penalty
        # -----------------------
        crash_penalty = -5.0 if info.get("crashed", False) else 0.0

        # -----------------------
        # Combined reward
        # -----------------------
        new_reward = base_reward + 0.6*angle_reward + 1.2*gap_reward + crash_penalty

        # Insert debugging values into info
        info["angle_diff"] = float(angle_diff)
        info["min_gap"] = float(min_dist)

        return obs, new_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

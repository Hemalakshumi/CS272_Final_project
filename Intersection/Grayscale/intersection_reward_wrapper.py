#intersection_reward_wrapper.py
import numpy as np
from gymnasium import Wrapper

class IntersectionRewardWrapper(Wrapper):
    """
    Clean, stable reward shaping for intersection-v0.
    Keeps reward scale small (0–5), avoids reward explosion.
    """

    def __init__(self, env):
        super().__init__(env)
        self.crash_penalty = -8.0

    def step(self, action):
        obs, base_r, terminated, truncated, info = self.env.step(action)
        env_u = self.env.unwrapped
        ego = env_u.vehicle

        # -------------------------
        # 1) Crash penalty
        # -------------------------
        crash_r = self.crash_penalty if info.get("crashed", False) else 0.0

        # -------------------------
        # 2) Heading alignment (small shaping)
        # -------------------------
        heading = ego.heading
        target = 0.0
        angle_diff = abs((heading - target + np.pi) % (2*np.pi) - np.pi)
        angle_r = 1.0 - (angle_diff / np.pi)      # in range ~0–1

        # -------------------------
        # 3) Safe distance reward
        # -------------------------
        min_dist = np.inf
        for v in env_u.road.vehicles:
            if v is ego: continue
            d = np.linalg.norm(v.position - ego.position)
            min_dist = min(min_dist, d)

        gap_r = np.tanh(min_dist / 8.0)           # 0–1 range

        # -------------------------
        # Final reward
        # -------------------------
        rew = base_r + 0.4*angle_r + 0.6*gap_r + crash_r

        # For debugging
        info["angle_diff"] = float(angle_diff)
        info["min_gap"] = float(min_dist)

        return obs, float(rew), terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


       

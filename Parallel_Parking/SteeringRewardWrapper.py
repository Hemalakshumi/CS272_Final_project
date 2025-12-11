from gymnasium import Wrapper
import numpy as np

class SteeringRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        # Now self.env is the underlying environment, so you can access self.env.vehicle

        x, y = self.env.vehicle.position
        extra = 0.0


        large_x_min, large_x_max = 12, 19
        large_y_min, large_y_max = 7, 9

        # Tight “alignment” zone
        tight_x_min, tight_x_max = 14, 18
        tight_y_min, tight_y_max = 7, 9


        if large_x_min <= x <= large_x_max and large_y_min <= y <= large_y_max:
            dist = np.linalg.norm([x - 16, y - 8])
            extra += 1.5 * np.exp(-(dist / 3.0)**2)


        if tight_x_min <= x <= tight_x_max and tight_y_min <= y <= tight_y_max:
            curr_heading = self.env.vehicle.heading
            goal_heading = self.env.goal_heading
            angle_diff = abs((curr_heading - goal_heading + np.pi) % (2*np.pi) - np.pi)
            angle_reward = np.exp(-(angle_diff / 0.3)**2)
            extra += 2.0 * angle_reward


        if base_reward==10:
            extra=extra+100

        if base_reward==20:
            extra=extra+200

        new_reward = base_reward + extra

        return obs, new_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)



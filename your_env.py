import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane
from highway_env.envs.common.observation import LidarObservation
from gymnasium.envs.registration import register


class YourEnv(AbstractEnv):

    def default_config(self):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "LidarObservation",
                "num_beams": 36,
                "maximum_range": 50,
                "normalize": True,
                "clip": False
            },
            "action": {"type": "ContinuousAction"},
            "duration": 200,
            "collision_reward": -5.0,
            "success_reward": 10.0,
            "out_of_bounds_penalty": -5.0,
            "delay_per_step": -0.1,
            "goal_tolerance": 0.4,
            "near_goal_position_reward": 2.0
        })
        return config


    # ---------------- RESET ----------------
    def _reset(self):
        net = RoadNetwork()
        lane_width = 4
        L = 40

        net.add_lane("a","b", StraightLane([0,0],  [L,0],  width=lane_width))
        net.add_lane("a","b", StraightLane([0,4],  [L,4],  width=lane_width))
        net.add_lane("a","b", StraightLane([0,8],  [L,8],  width=lane_width))

        self.road = Road(network=net, np_random=self.np_random)
        self.road.vehicles = []

        # ------ Parked cars (top lane y=8) ------
        for x in [0,8,24]:
            if 14 <= x <= 20:  
                continue
            car = Vehicle(self.road, position=[x, 8], heading=0, speed=0)
            car.is_controlled = False
            self.road.vehicles.append(car)

        # ------ Ego in middle lane ------
        ego = Vehicle(self.road, position=[6,4], heading=0, speed=0)
        ego.color = (255,0,0)
        self.vehicle = ego
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)

        # Goal inside parking gap
        self.goal_position = np.array([17, 8])
        self.goal_heading = 0

        # Create LidarObservation correctly
        self.observation_type = LidarObservation(self)

        self.steps = 0
        return self.observation_type.observe()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._reset()
        return obs, {}

    # ---------------- REWARD ----------------
    def _reward(self, action):
        distance = np.linalg.norm(self.vehicle.position - self.goal_position)

        # crash
        if self.vehicle.crashed:
            return -5

        reward = -0.3 * (distance / 30)  # shaping

        # success
        if distance < self.config["goal_tolerance"]:
            reward += self.config["success_reward"]

        # step penalty
        reward += self.config["delay_per_step"]

        return float(reward)

    # ---------------- TERMINALS ----------------
    def _is_terminal(self):
        distance = np.linalg.norm(self.vehicle.position - self.goal_position)
        if self.vehicle.crashed:
            return True
        return distance < self.config["goal_tolerance"]

    def _is_truncated(self):
        return self.steps >= self.config["duration"]

    # ---------------- STEP ----------------
    def _step(self, action):
        self.vehicle.act(action)
        self.road.step(1.0)
        self.steps += 1

        obs = self.observation_type.observe()
        reward = self._reward(action)
        done = self._is_terminal()
        truncated = self._is_truncated()

        return obs, reward, done, truncated, {}


# ----- Register -----
register(
    id="CustomParallelEnv-v0",
    entry_point="your_env:YourEnv",
    max_episode_steps=150,
)

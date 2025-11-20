Parallel Parking Environment (ParallelParking-v0)
Overview

Our custom environment ParallelParking-v0 simulates a simplified parallel-parking scenario built on top of Highway-Env and Gymnasium.
This environment is designed as an RL benchmark for continuous-control parking tasks with shaped rewards and goal-oriented observations.

Key Customizations

A 3-lane road where the top lane contains two parked cars, forming a parallel-parking region.

The ego-vehicle starts in the middle lane and must maneuver into the parking slot.

The goal position is fixed at (16, 8) with a required orientation of 0°.

A custom reward function encourages smooth, accurate parking without collisions.

Uses KinematicsGoalObservation, providing ego + goal features for stable training.

Objective

The agent (ego-car) must learn to:

Navigate from the middle lane into the parking slot between parked cars.

Avoid collisions with parked vehicles and road boundaries.

Approach the goal position with correct alignment.

Stay within road boundaries at all times.

Efficiently maneuver into the slot (reward shaping encourages proper approach path and angle).

Observation & Action Spaces
Observation

The environment uses:

KinematicsGoalObservation

With the following configuration:

Feature	Description
x, y	Ego position
vx, vy	Ego velocity
cos_h, sin_h	Ego orientation
+ Goal features	normalized

Normalized using scales:
[50, 10, 10, 10, 1, 1]

Action

Uses:

ContinuousAction
(typically steering + acceleration)

Reward Function

Your environment uses the following reward components:

Reward	Purpose
collision_reward = -5	Penalize hitting parked cars or going off lane
out_of_bounds_penalty = -5	Penalize leaving the road boundary
success_reward = 10	Reward for reaching the goal AND correct heading
near_goal_position_reward = 2.0	Reward for being close to goal
near_goal_reward = 3.0	Bonus for being inside parking bounding box
additional_alignment_reward = 2.0	Extra reward when perfectly aligned
Distance shaping	Smooth negative shaping based on distance to goal

This reward structure enables stable gradients for policies like SAC / PPO.

Termination & Truncation
Episode Terminates If:

The ego-vehicle collides with another vehicle.

The agent successfully parks (goal position + heading threshold).

Episode Truncates If:

Step count exceeds duration = 200.

Vehicle leaves the road (x > 24 or y > 10).

Environment Registration

Add this to highway_env/envs/__init__.py:

from gymnasium.envs.registration import register

register(
    id="ParallelParking-v0",
    entry_point="highway_env.envs.parallel_parking_env:ParallelParkingEnv",
)

Usage Example
import gymnasium as gym
import highway_env

env = gym.make("ParallelParking-v0", render_mode="rgb_array")

obs, info = env.reset()

for step in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()

Installation
pip install highway-env gymnasium numpy

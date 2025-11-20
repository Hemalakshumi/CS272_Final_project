# Parallel Parking Environment (ParallelParking-v0)

## Overview

This custom environment, **ParallelParking-v0**, simulates a simplified parallel-parking scenario built on top of **Highway-Env** and **Gymnasium**. It is intended as an RL benchmark for continuous-control parking tasks using shaped rewards and goal-oriented observations.

## Key customizations

- A 3-lane road where the top lane contains two parked cars, creating a parallel-parking region.
- The ego vehicle starts in the middle lane and must maneuver into the parking slot.
- The goal position is fixed at **(16, 8)** with a required orientation of **0°**.
- A custom reward function encourages smooth, accurate parking without collisions.
- Uses `KinematicsGoalObservation` (ego + goal features) for stable training.

## Objective

The agent (ego car) must learn to:

- Navigate from the middle lane into the parking slot between parked cars.
- Avoid collisions with parked vehicles and road boundaries.
- Approach the goal position with correct alignment.
- Stay within road boundaries at all times.
- Efficiently maneuver into the slot (reward shaping encourages proper approach path and angle).

## Observation & action spaces

### Observation

- Observation type: `KinematicsGoalObservation`.
- Features and descriptions:

  - `x, y` — ego position
  - `vx, vy` — ego velocity
  - `cos_h, sin_h` — ego orientation
  - goal features — normalized

- Normalization scales: `[50, 10, 10, 10, 1, 1]`.

### Action

- Action type: `ContinuousAction` (typically `[steering, acceleration]`).

## Reward function

The environment uses the following reward components (defaults shown):

- `collision_reward = -5` — penalize hitting parked cars or going off lane
- `out_of_bounds_penalty = -5` — penalize leaving the road boundary
- `success_reward = 10` — reward for reaching the goal and correct heading
- `near_goal_position_reward = 2.0` — reward for being close to the goal
- `near_goal_reward = 3.0` — bonus for being inside the parking bounding box
- `additional_alignment_reward = 2.0` — extra reward when well aligned
- Distance shaping — smooth negative shaping based on distance to goal

This structure provides smoother gradients for policy optimization algorithms such as SAC or PPO.

For exact reward calculations, see the `_reward()` implementation in `custom_env.py`.

## Termination & truncation

### Episode termination

- The ego vehicle collides with another vehicle (immediate termination).
- The agent successfully parks (goal position within tolerance and heading threshold).

### Episode truncation

- Step count exceeds the duration (default `duration = 200`).
- Vehicle leaves the road (`x > 24` or `y > 10`).

## Environment registration

Add the following to `highway_env/envs/__init__.py` to register the environment:

```python
from gymnasium.envs.registration import register

register(
    id="ParallelParking-v0",
    entry_point="highway_env.envs.parallel_parking_env:ParallelParkingEnv",
)
```

## Usage example

```python
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
```

## Installation

Install dependencies with pip:

```bash
pip install highway-env gymnasium numpy
```

# Highway Environment - Reinforcement Learning Project

This folder contains a complete reinforcement learning implementation using **Proximal Policy Optimization (PPO)** to train autonomous agents in the Highway-Env environment. The project includes both grayscale image-based observation and LIDAR-based approaches.

## Project Structure

```
Highway/
├── Grayscale/          # CNN-based approach using grayscale observations
│   ├── train_highway.py
│   ├── test_highway2.py
│   ├── custom_ppo.py
│   ├── custom_cnn.py
│   ├── custom_rollout_buffer.py
│   ├── plot_learningCurve.py
│   ├── plot_500episodes_violin.py
│   ├── monitor.csv
│   └── README.md
├── Lidar/              # LIDAR-based approach (empty - placeholder)
├── logs/               # TensorBoard logs from training runs (PPO_1 through PPO_6)
└── videos/             # Recording of agent gameplay during evaluation
```

## Overview

### Grayscale Approach

This implementation trains a PPO agent to navigate a highway environment using **grayscale image observations** processed through a custom CNN feature extractor.

## Core Components

### 1. **train_highway.py**
The main training script that initializes and trains the PPO agent.

**Key Features:**
- Configures the highway environment with grayscale observations (128×64 pixels, 4-frame stack)
- Uses `CustomPPO` with a `CustomCNN` feature extractor
- Training hyperparameters:
  - Learning rate: 3e-4
  - Steps per rollout: 1024
  - Batch size: 64
  - Epochs per update: 10
  - Discount factor (gamma): 0.99
  - Total training timesteps: 200,000
- Logs training metrics to TensorBoard (`./logs/`)
- Saves trained model as `ppo_highway_grayscale_meta`

**Usage:**
```bash
python train_highway.py
```

### 2. **custom_cnn.py**
Implements a custom CNN feature extractor for processing grayscale image observations.

**Architecture:**
```
Input (128×64 grayscale, 4 frames stacked)
  ↓
Conv2d(4→32, kernel=8, stride=4) + ReLU
  ↓
Conv2d(32→64, kernel=4, stride=2) + ReLU
  ↓
Conv2d(64→64, kernel=3, stride=1) + ReLU
  ↓
Flatten + Linear(n_flatten→512) + ReLU
  ↓
Output (512-dim feature vector)
```

**Key Class:**
- `CustomCNN(BaseFeaturesExtractor)`: Extracts visual features from grayscale images for the policy network

### 3. **custom_ppo.py**
Extends Stable Baselines3's PPO implementation with customizations for the highway task.

**Customizations:**
- Integrates `CustomRolloutBuffer` for enhanced data collection
- Implements the standard PPO training loop with:
  - Advantage normalization
  - Clipped objective function
  - Policy gradient estimation
- Handles discrete action spaces (left, right, accelerate, brake)

**Key Class:**
- `CustomPPO(PPO)`: Modified PPO algorithm with custom reward shaping and data handling

### 4. **custom_rollout_buffer.py**
Custom buffer that extends Stable Baselines3's `RolloutBuffer` to store additional transition information.

**Features:**
- Stores environment `info` dictionaries alongside transitions
- Methods:
  - `add()`: Adds transitions with optional metadata
  - `get_infos()`: Retrieves infos for batches of transitions

**Key Class:**
- `CustomRolloutBuffer(RolloutBuffer)`: Enhanced buffer with info tracking for debugging and analysis

### 5. **test_highway2.py**
Evaluation script that tests the trained model and records gameplay videos.

**Features:**
- Loads the trained `ppo_highway_grayscale_meta` model
- Runs 500 evaluation episodes
- Records videos every 50 episodes to the `videos/` folder
- Collects episode rewards for statistical analysis
- Configuration matches training environment

**Usage:**
```bash
python test_highway2.py
```

### 6. **plot_learningCurve.py**
Generates a learning curve visualization from training logs.

**Features:**
- Reads `monitor.csv` (logged by Stable Baselines3 Monitor wrapper)
- Plots both raw episode rewards and smoothed curve (50-episode window)
- Saves visualization to `learning_curve.png`

**Usage:**
```bash
python plot_learningCurve.py
```

### 7. **plot_500episodes_violin.py**
Creates a violin plot of evaluation episode rewards.

**Features:**
- Loads `test_rewards_500.npy` (saved reward array from test_highway2.py)
- Generates violin plot showing reward distribution
- Saves visualization to `violin_500.png`

**Usage:**
```bash
python plot_500episodes_violin.py
```

## Environment Configuration

The Highway-Env is configured with the following observation and action spaces:

**Observation:**
- Type: Grayscale (4-channel stacked frames)
- Shape: 128×64 pixels per frame
- Preprocessing: RGB to grayscale using standard weights [0.2989, 0.5870, 0.1140]
- Scaling: 1.75× magnification for better detail

**Action Space:**
- Type: Discrete (4 actions)
- Actions: SLOWER, IDLE, FASTER, LEFT, RIGHT (via DiscreteMetaAction)

**Episode Settings:**
- Policy frequency: 2 (agent decides every 2 steps)

## Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Learning Rate | 3e-4 |
| Discount Factor (γ) | 0.99 |
| GAE Lambda | 0.95 (default) |
| Entropy Coefficient | 0.0 (default) |
| Steps per Rollout | 1024 |
| Batch Size | 64 |
| Epochs per Update | 10 |
| Total Timesteps | 200,000 |
| Feature Dimension | 512 |

### Outputs

- **Model**: `ppo_highway_grayscale_meta` (trained weights)
- **Logs**: `./logs/PPO_1/` through `./logs/PPO_6/` (TensorBoard event files)
- **Monitoring**: `monitor.csv` (episode rewards and lengths)

## Evaluation

The trained agent is evaluated over 500 episodes to assess performance:

- Recorded videos show the agent's decision-making and driving behavior
- Reward statistics provide quantitative performance metrics
- Learning curve visualizations track improvement over training

## Dependencies

```
gymnasium
highway-env
stable-baselines3
torch
numpy
pandas
matplotlib
opencv-python
```

## Quick Start

1. **Train a model:**
   ```bash
   python train_highway.py
   ```

2. **Evaluate the model:**
   ```bash
   python test_highway2.py
   ```

3. **Generate learning curve:**
   ```bash
   python plot_learningCurve.py
   ```

4. **Generate evaluation statistics:**
   ```bash
   python plot_500episodes_violin.py
   ```

## File Descriptions Summary

| File | Purpose |
|------|---------|
| `train_highway.py` | Main training script |
| `test_highway2.py` | Evaluation and video recording |
| `custom_ppo.py` | Custom PPO implementation |
| `custom_cnn.py` | CNN feature extractor for image observations |
| `custom_rollout_buffer.py` | Custom data buffer for training |
| `plot_learningCurve.py` | Visualization of learning progress |
| `plot_500episodes_violin.py` | Statistical visualization of evaluation results |
| `monitor.csv` | Training episode metrics |
| `test_rewards_500.npy` | Evaluation episode rewards (numpy array) |

## Future Work

- **LIDAR Approach**: Implement and complete the LIDAR-based observation method in the `Lidar/` folder
- **Hyperparameter Tuning**: Experiment with different learning rates, network architectures, and training lengths
- **Comparative Analysis**: Compare grayscale CNN approach with LIDAR-based approach
- **Model Improvement**: Implement reward shaping to encourage safer driving behaviors

## Notes

- Multiple training runs (PPO_1 through PPO_6) are logged in the `logs/` folder
- Videos of agent gameplay are saved in the `videos/` folder during evaluation
- All training uses Stable Baselines3 and PyTorch for deep learning components

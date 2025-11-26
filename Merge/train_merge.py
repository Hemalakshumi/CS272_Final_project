# train_merge_dqn.py

import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from custom_cnn import CustomCNN

env_config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 2,
    "vehicles_count": 20,
    "policy_frequency": 2,
}

env = gym.make("merge-v0", config=env_config)
obs, info = env.reset()

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512)
)

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=200_000,
    batch_size=64,
    learning_starts=5000,
    gamma=0.99,
    train_freq=4,
    target_update_interval=500,
    exploration_fraction=0.25,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./logs/",
    verbose=1,
)

model.learn(200_000)
model.save("dqn_merge_grayscale")
env.close()

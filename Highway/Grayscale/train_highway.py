import highway_env
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor

from custom_ppo import CustomPPO
from custom_cnn import CustomCNN

config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],
        "scaling": 1.75,
    },
    "action": {"type": "DiscreteMetaAction"},
    "policy_frequency": 2,
}

env = gym.make("highway-v0", config=config)
env = Monitor(env, filename="monitor.csv")  

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

model = CustomPPO(
    "CnnPolicy",
    env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./logs/",
)

model.learn(200_000)

model.save("ppo_highway_grayscale_meta")
env.close()

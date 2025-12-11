import os
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_schedule_fn
from custom_cnn import CustomCNN
from make_env import create_vec_env

# ==========================
# ENV CONFIG
# ==========================
env_config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],
        "scaling": 1.75,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 2,
    "vehicles_count": 30,
    "policy_frequency": 2,
}

os.makedirs("./monitor", exist_ok=True)

# Build vec env with monitor enabled
vec_env = create_vec_env(env_config, monitor_dir="./monitor")

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

model = PPO(
    "CnnPolicy",
    vec_env,
    policy_kwargs=policy_kwargs,
    learning_rate=get_schedule_fn(3e-4),
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    ent_coef=0.01,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./merge_logs/",
)

model.learn(200_000)
model.save("ppo_merge_wrapped")
vec_env.save("merge_vecnorm.pkl")

print("Training completed.")

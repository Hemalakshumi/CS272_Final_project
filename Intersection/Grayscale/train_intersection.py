# train_intersection.py
import os
import gymnasium as gym
import highway_env

from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from custom_cnn import CustomCNN   
from make_intersection_env import make_intersection_env_fn
from intersection_reward_wrapper import IntersectionRewardWrapper

# -------------------------
# Config
# -------------------------
env_config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],
        "scaling": 1.75,
    },
    "action": {"type": "DiscreteMetaAction"},
    "duration": 120,
    "policy_frequency": 2,
    "simulation_frequency": 15,
}

logdir = "./ppo_intersection_logs"
os.makedirs(logdir, exist_ok=True)
monitor_dir = os.path.join(logdir, "monitor")
os.makedirs(monitor_dir, exist_ok=True)
os.makedirs("./intersection_checkpoints", exist_ok=True)
os.makedirs("./intersection_best_model", exist_ok=True)
os.makedirs("./intersection_eval_logs", exist_ok=True)


def make_raw_env():
    def _init():
        env = make_intersection_env_fn(env_config, render_mode=None)()
        # Apply the reward wrapper INSIDE (so Monitor still outermost)
        env = IntersectionRewardWrapper(env)
        # TimeLimit is handled in the factory used by make_intersection_env_fn if needed
        return env
    return _init

# -------------------------
# Training VecEnv
# -------------------------
def make_train_env(rank):
    def _init():
        raw = make_raw_env()()
        monitor_path = os.path.join(monitor_dir, f"monitor_{rank}.csv")
        env = Monitor(raw, filename=monitor_path) 
        return env
    return _init

n_envs = 8
train_vec = DummyVecEnv([make_train_env(i) for i in range(n_envs)])
train_vec = VecNormalize(train_vec, norm_obs=True, norm_reward=True)

# -------------------------
# Policy + model
# -------------------------
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
    normalize_images=False,
)

lr_schedule = get_schedule_fn(3e-4)

model = PPO(
    "CnnPolicy",
    train_vec,
    policy_kwargs=policy_kwargs,
    learning_rate=lr_schedule,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=logdir,
)

# -------------------------
# Eval env: must be same type as training (VecNormalize)
# Build raw eval env -> Monitor -> DummyVecEnv -> VecNormalize(training=False)
# -------------------------
raw_eval = make_raw_env()()
raw_eval = Monitor(raw_eval, filename=os.path.join("./intersection_eval_logs", "eval_monitor.csv"))
eval_dummy = DummyVecEnv([lambda: raw_eval])
eval_vec = VecNormalize(eval_dummy, training=False, norm_reward=False)
eval_vec.obs_rms = train_vec.obs_rms

# -------------------------
# Callbacks: Checkpoints + Eval
# -------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./intersection_checkpoints/",
    name_prefix="ppo_intersection"
)

eval_callback = EvalCallback(
    eval_vec,
    best_model_save_path="./intersection_best_model/",
    log_path="./intersection_eval_logs/",
    eval_freq=50_000,
    n_eval_episodes=8,
    deterministic=True,
)

# -------------------------
# Train
# -------------------------
total_timesteps = 200_000
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

# Save final model and VecNormalize stats
model.save(os.path.join(logdir, "ppo_intersection_final"))
train_vec.save("intersection_vecnorm.pkl")
print("Training finished and saved.")

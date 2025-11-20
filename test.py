import time
import gymnasium as gym
from stable_baselines3 import PPO
import your_env   # registers env

def test():
    # Render mode MUST be passed here
    env = your_env.YourEnv(render_mode="human")

    model = PPO.load("ppo_parking_short")

    obs, _ = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.05)   # Slow down for visualization

    print("Episode finished.")

if __name__ == "__main__":
    test()

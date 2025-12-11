from gymnasium.wrappers import TimeLimit
import gymnasium as gym

def make_intersection_env_fn(config, render_mode=None, eval_mode=False):
    """
    Return a factory that creates a raw intersection-v0 environment.
    
    """
    def _init():
        env = gym.make("intersection-v0", config=config, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=config.get("duration", 200))
        return env
    return _init
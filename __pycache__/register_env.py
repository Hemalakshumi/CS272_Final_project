from gymnasium.envs.registration import register

register(
    id="YourParkingEnv-v0",
    entry_point="your_env:YourEnv",   # fileName:ClassName
)

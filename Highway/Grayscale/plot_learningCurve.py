import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("monitor.csv", skiprows=1)

# Check columns
print(df.columns)

# SB3 format:
# r = reward
# l = episode length
# t = total timesteps

episodes = range(len(df))

plt.figure(figsize=(8, 5))
plt.plot(episodes, df["r"], alpha=0.3, label="Raw reward")
plt.plot(episodes, df["r"].rolling(50).mean(), label="Smoothed (window=50)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Learning Curve")
plt.grid()
plt.legend()
plt.savefig("learning_curve.png", dpi=300)
plt.show()

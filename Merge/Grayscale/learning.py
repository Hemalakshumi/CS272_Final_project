import pandas as pd
import matplotlib.pyplot as plt
import glob

# -----------------------------------------------------
# 1. LOAD ALL MONITOR LOG FILES
# -----------------------------------------------------

monitor_files = glob.glob("monitor/monitor_*.csv.monitor.csv")

print("Found monitor files:")
for f in monitor_files:
    print("  ", f)

# Read and combine all logs
dfs = [pd.read_csv(f, skiprows=1) for f in monitor_files]
df = pd.concat(dfs, ignore_index=True)

# Sort by time (optional but recommended)
df = df.sort_values("t")

# -----------------------------------------------------
# 2. EXTRACT REWARDS
# -----------------------------------------------------

rewards = df["r"].values

# Moving average smoothing
window = 50
smooth = pd.Series(rewards).rolling(window).mean()

# -----------------------------------------------------
# 3. PLOT LEARNING CURVE
# -----------------------------------------------------

plt.figure(figsize=(14, 6))
plt.plot(rewards, alpha=0.30, label="Raw Episode Rewards")
plt.plot(smooth, color="orange", linewidth=2, label=f"Smoothed (window={window})")

plt.title("PPO Training — MERGE Environment Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.legend()
plt.grid(True)

plt.show()

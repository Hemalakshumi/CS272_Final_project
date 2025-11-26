import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/PPO_0.monitor.csv")

plt.plot(df["l"], df["r"])
plt.xlabel("Training Step")
plt.ylabel("Reward")
plt.title("Learning Curve")
plt.grid()
plt.savefig("learning_curve.png")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("test_rewards_500.npy")

plt.figure(figsize=(6, 6))
plt.violinplot([rewards], showmeans=True)
plt.title("500 Episode Evaluation")
plt.ylabel("Reward")
plt.xticks([1], ["PPO"])
plt.savefig("violin_500.png", dpi=300)
plt.show()

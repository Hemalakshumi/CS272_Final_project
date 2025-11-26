import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("test_rewards_500.npy")

plt.violinplot(rewards, showmeans=True)
plt.title("500 Episode Evaluation")
plt.ylabel("Reward")
plt.savefig("violin_500.png")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from test_merge import rewards
# ---- VIOLIN PLOT FOR TEST EPISODE REWARDS ----

plt.figure(figsize=(10, 6))

sns.violinplot(
    data=rewards,
    inner="quartile",     # shows Q1, median, Q3
    color="skyblue"
)

plt.title("Distribution of Test Episode Rewards (Violin Plot)", fontsize=14)
plt.ylabel("Reward", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "model_checkpoints_lidar_optimized"

TRAIN_LOG_PATH = os.path.join(BASE_DIR, "logs", "intersection_training_rewards.npy")
TEST_REWARDS_PATH = os.path.join(BASE_DIR, "logs", "intersection_test_rewards_optimized.txt")
PLOTS_DIR = os.path.join(BASE_DIR, "plots", "intersection_optimized")

os.makedirs(PLOTS_DIR, exist_ok=True)


def moving_average(x, window=50):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return x
    if len(x) < window:
        window = len(x)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    pad = np.full(window - 1, ma[0], dtype=np.float32)
    return np.concatenate([pad, ma], axis=0)


def plot_learning_curve():
    if not os.path.exists(TRAIN_LOG_PATH):
        raise FileNotFoundError(f"Training log not found at {TRAIN_LOG_PATH}")

    rewards = np.load(TRAIN_LOG_PATH).astype(np.float32).reshape(-1)
    print(f"Loaded training rewards from {TRAIN_LOG_PATH} with shape {rewards.shape}")

    if rewards.size == 0:
        print("No training rewards found.")
        return

    episodes = np.arange(1, len(rewards) + 1)
    smoothed = moving_average(rewards, window=50)

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, rewards, alpha=0.3, label="Raw episodic return")
    plt.plot(episodes, smoothed, label="Moving average (window=50)")
    plt.xlabel("Episode")
    plt.ylabel("Episodic return")
    plt.title("Intersection-v0 LiDAR – PPO Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(PLOTS_DIR, "intersection_optimized_learning_curve.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved learning curve: {out_path}")


def plot_test_violin():
    if not os.path.exists(TEST_REWARDS_PATH):
        raise FileNotFoundError(f"Test rewards file not found at {TEST_REWARDS_PATH}")

    test_rewards = np.loadtxt(TEST_REWARDS_PATH, dtype=np.float32).reshape(-1)
    print(f"Loaded test rewards from {TEST_REWARDS_PATH} with shape {test_rewards.shape}")

    plt.figure(figsize=(6, 5))
    plt.violinplot(test_rewards, showmeans=True, showextrema=True)
    plt.xticks([1], ["Intersection PPO (LiDAR)"])
    plt.ylabel("Episodic return")
    plt.title("Intersection-v0 LiDAR – PPO Test Performance (500 Episodes)")
    plt.grid(True, axis="y", alpha=0.3)

    out_path = os.path.join(PLOTS_DIR, "intersection_optimized_test_violin.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved test violin plot: {out_path}")


if __name__ == "__main__":
    plot_learning_curve()
    plot_test_violin()

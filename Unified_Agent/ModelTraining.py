!pip install 'shimmy>=2.0'

# Create environment
env = UnifiedDrivingEnv()

# Initialize DQN model
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    buffer_size=15000,
    learning_starts=200,
    batch_size=32,
    target_update_interval=1000,
    verbose=1,
)

# Train for 50,000 timesteps
model.learn(total_timesteps=50_000)

# Save the trained model
model.save("dqn_unified_driving_model")

print("Model saved as 'dqn_unified_driving_model.zip'")

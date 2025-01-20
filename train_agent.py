from stable_baselines3 import PPO
from trading_env import TradingEnv
import pandas as pd
import os

# Load stock data
data = pd.read_csv("AAPL_data.csv", index_col="date", parse_dates=True)

# Initialize the environment
env = TradingEnv(data)

# Path to the pre-trained model
model_path = "./ppo_trading_agent.zip"

# Load pre-trained model if it exists
if os.path.exists(model_path):
    print(f"Loading pre-trained model from {model_path}...")
    model = PPO.load(model_path, env)
else:
    print("No pre-trained model found. Creating a new model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.00005,
        gamma=0.995,
        clip_range=0.1,
    )

# Train the model
print("Training the PPO model...")
model.learn(total_timesteps=300000)  # Increased timesteps
print("Training completed!")

# Save the model
model.save(model_path)
print(f"Model saved at {model_path}.")

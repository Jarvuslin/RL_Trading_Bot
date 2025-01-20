import os
import pandas as pd
from trading_env import TradingEnv
from stable_baselines3 import PPO

# Load stock data
data = pd.read_csv("AAPL_data.csv", index_col="date", parse_dates=True)

# Initialize the environment
env = TradingEnv(data)

# Path to the saved model
model_path = "./ppo_trading_agent.zip"

# Load the trained model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please run train_agent.py first.")
print(f"Loading model from {model_path}...")
model = PPO.load(model_path)
print("Model loaded successfully!")

# Test the model
obs, info = env.reset()
terminated = False
truncated = False
total_rewards = []
total_assets = []
steps = []

while not (terminated or truncated):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_rewards.append(reward)
    total_assets.append(env.balance + (env.shares * data.iloc[env.current_step]['Close']))
    steps.append(env.current_step)

# Save results
results = pd.DataFrame({"Step": steps, "Total Assets": total_assets, "Reward": total_rewards})
results.to_csv("test_results.csv", index=False)
print("Testing completed. Results saved to 'test_results.csv'.")

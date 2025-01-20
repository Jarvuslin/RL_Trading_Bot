import pandas as pd
from trading_env import TradingEnv

# Load your data (ensure it's preprocessed and saved as a CSV)
data = pd.read_csv("AAPL_data.csv", index_col="date", parse_dates=True)

# Initialize the environment
env = TradingEnv(data)

# Test the environment
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Random action for testing
    obs, reward, done, info = env.step(action)
    env.render()

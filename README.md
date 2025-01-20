
# RL Trading Bot

An AI-powered trading bot built using **Reinforcement Learning (RL)**. The bot is trained using the **Proximal Policy Optimization (PPO)** algorithm to make buy, sell, or hold decisions in a simulated trading environment. This project uses historical stock data to train and test the trading agent.

---

## Features

- **Custom Trading Environment**: Designed using `gymnasium` with support for financial indicators like moving averages and RSI.
- **Reinforcement Learning**: Implements the PPO algorithm from `stable-baselines3` for decision-making.
- **Data Source**: Supports stock data fetched using APIs (e.g., Alpha Vantage) or external CSV files.
- **Visualization**: Generates performance plots for total assets and rewards over time.
- **Modular Design**: Easy to replace or extend components like the dataset, model, or environment.

---

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- A package manager like `pip`

Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Required Libraries

- `gymnasium`
- `numpy`
- `pandas`
- `stable-baselines3`
- `matplotlib`
- `yfinance` (if using Yahoo Finance for data fetching)
- `alpha_vantage` (optional, if using Alpha Vantage API)

---

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo-url/rl-trading-bot.git
cd rl-trading-bot
```

---

## Code

### Fetch Stock Data (`fetch_data.py`)

```python
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

API_KEY = "YOUR_API_KEY"

def fetch_stock_data(symbol="AAPL", output_file="AAPL_data.csv"):
    ts = TimeSeries(key=API_KEY, output_format="pandas")
    data, meta_data = ts.get_daily(symbol=symbol, outputsize="full")
    data.to_csv(output_file)
    print(f"Data saved to {output_file}")

fetch_stock_data()
```

If you want to use Yahoo Finance instead:
```python
import yfinance as yf

def fetch_stock_data(symbol="AAPL", output_file="AAPL_data.csv"):
    data = yf.download(symbol, start="2000-01-01")
    data.to_csv(output_file)
    print(f"Data saved to {output_file}")

fetch_stock_data()
```

---

### Custom Trading Environment (`trading_env.py`)

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares = 0

        # Add technical indicators
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['RSI'] = self._calculate_rsi(self.data['Close'])
        self.data = self.data.dropna()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.data.columns),), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        return self._get_observation(), {}

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        reward = 0

        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            self.shares += shares_to_buy
            self.balance -= shares_to_buy * current_price
        elif action == 2:  # Sell
            self.balance += self.shares * current_price
            self.shares = 0

        total_assets = self.balance + (self.shares * current_price)
        reward = ((total_assets - self.initial_balance) / 1e3)

        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        current_price = self.data.iloc[self.current_step]['Close']
        total_assets = self.balance + (self.shares * current_price)
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares}, Total Assets: {total_assets:.2f}")

    def _get_observation(self):
        obs = self.data.iloc[self.current_step].values / self.data.iloc[0].values
        return obs

    @staticmethod
    def _calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
```

---

### Train the Agent (`train_agent.py`)

```python
from stable_baselines3 import PPO
from trading_env import TradingEnv
import pandas as pd

data = pd.read_csv("AAPL_data.csv", index_col="date", parse_dates=True)
env = TradingEnv(data)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.00005, gamma=0.995, clip_range=0.1)
model.learn(total_timesteps=300000)
model.save("./ppo_trading_agent.zip")
```

---

### Test the Agent (`test_agent.py`)

```python
from stable_baselines3 import PPO
from trading_env import TradingEnv
import pandas as pd

data = pd.read_csv("AAPL_data.csv", index_col="date", parse_dates=True)
env = TradingEnv(data)

model = PPO.load("./ppo_trading_agent.zip")
obs, info = env.reset()
terminated, truncated = False, False

while not (terminated or truncated):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```

---

### Plot Results (`plot_results.py`)

```python
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("test_results.csv")

plt.figure(figsize=(10, 6))
plt.plot(results["Step"], results["Total Assets"], label="Total Assets")
plt.title("Agent Performance Over Time")
plt.xlabel("Step")
plt.ylabel("Total Assets")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(results["Step"], results["Reward"], label="Reward", color="orange")
plt.title("Rewards Over Time")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.legend()
plt.grid()
plt.show()
```

---

## Results

- **Agent Performance Over Time**: Shows the growth of total assets as the agent trades.
- **Rewards Over Time**: Tracks the agent's rewards for each step.

---

## Future Improvements

- Integrate live trading functionality with APIs like Alpaca or Interactive Brokers.
- Add support for multiple stocks and portfolio management.
- Implement advanced RL algorithms (e.g., DDPG, SAC).

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """
    Custom Trading Environment for Reinforcement Learning.
    """

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
        self.data = self.data.dropna()  # Drop rows with NaN values

        # Define action and observation space
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

        # Perform the action
        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            self.shares += shares_to_buy
            self.balance -= shares_to_buy * current_price
        elif action == 2:  # Sell
            self.balance += self.shares * current_price
            self.shares = 0

        # Calculate total assets
        total_assets = self.balance + (self.shares * current_price)

        # Reward calculation: scaled by total assets, penalizing risk and encouraging profit-locking
        reward = ((total_assets - self.initial_balance) / 1e3)

        # Risk management: Penalize holding too many shares compared to balance
        if self.shares > self.balance * 0.5:
            reward -= 5

        # Profit locking: Encourage securing gains
        if total_assets > self.initial_balance * 1.5:
            reward += 10

        # Stop-loss penalty
        if total_assets < self.initial_balance * 0.9:
            reward -= 10

        # Increment step
        self.current_step += 1

        # Check if we're at the end of the data
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

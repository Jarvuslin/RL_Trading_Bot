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

Required Libraries
gymnasium
numpy
pandas
stable-baselines3
matplotlib
yfinance (if using Yahoo Finance for data fetching)
alpha_vantage (optional, if using Alpha Vantage API)


etup
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-repo-url/rl-trading-bot.git
cd rl-trading-bot
2. Fetch Stock Data
Option 1: Alpha Vantage API
Create a file named fetch_data.py and use the following script to download stock data:

python
Copy
Edit
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

API_KEY = "YOUR_API_KEY"

def fetch_stock_data(symbol="AAPL", output_file="AAPL_data.csv"):
    ts = TimeSeries(key=API_KEY, output_format="pandas")
    data, meta_data = ts.get_daily(symbol=symbol, outputsize="full")
    data.to_csv(output_file)
    print(f"Data saved to {output_file}")

fetch_stock_data()
Option 2: Yahoo Finance
Install the yfinance library:

bash
Copy
Edit
pip install yfinance
Run the following script to fetch stock data:

python
Copy
Edit
import yfinance as yf

def fetch_stock_data(symbol="AAPL", output_file="AAPL_data.csv"):
    data = yf.download(symbol, start="2000-01-01")
    data.to_csv(output_file)
    print(f"Data saved to {output_file}")

fetch_stock_data()
Ensure the data is saved as AAPL_data.csv in the project directory.

Usage
1. Training the Agent
Train the PPO agent using the historical stock data:

bash
Copy
Edit
python train_agent.py
This will:

Train the agent for a specified number of timesteps.
Save the trained model as ppo_trading_agent.zip.
2. Testing the Agent
Evaluate the agent's performance:

bash
Copy
Edit
python test_agent.py
This will:

Load the trained model.
Test the agent using the historical stock data.
Save the results as test_results.csv.
3. Visualizing Results
Generate performance plots:

bash
Copy
Edit
python plot_results.py
This will:

Plot total assets over time.
Plot rewards over time.

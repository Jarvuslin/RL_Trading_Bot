import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from config import ALPHA_VANTAGE_API_KEY

def fetch_stock_data(symbol):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='full')
    data = data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    return data

if __name__ == "__main__":
    stock_data = fetch_stock_data("AAPL")
    stock_data.to_csv("AAPL_data.csv")

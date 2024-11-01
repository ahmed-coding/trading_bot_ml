# collect_data.py

import csv
import numpy as np
import statistics
from binance.client import Client
from utils import get_top_symbols  # Import get_top_symbols for consistency
from config import API_KEY,API_SECRET

# Initialize Binance client
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'

def collect_data(symbol, interval='1m', limit=100):
    """
    Collect short-term data for scalping strategy with 1-minute intervals.
    Saves data with indicators that capture rapid price movements.
    """
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    with open(f"data/{symbol}_scalping_data.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Open", "High", "Low", "Close", "Volume", "MA3", "MA5", "RSI", "Volatility", "Trend"])

        for i in range(5, len(klines)):
            close_prices = [float(k[4]) for k in klines[i-5:i]]
            ma3 = np.mean(close_prices[-3:])
            ma5 = np.mean(close_prices)
            rsi = 100 - (100 / (1 + np.mean(np.diff(close_prices) > 0) / np.mean(np.diff(close_prices) <= 0)))
            volatility = statistics.stdev(close_prices)
            trend = 1 if close_prices[-1] > close_prices[0] else 0  # 1 if price increased, 0 if decreased

            row = klines[i][:5] + [ma3, ma5, rsi, volatility, trend]
            writer.writerow(row)

if __name__ == "__main__":
    symbols = get_top_symbols(limit=10, profit_target=0.003)  # Automatically fetch top symbols
    for symbol in symbols:
        collect_data(symbol)
        print(f"Data collection complete for {symbol}.")

# collect_data.py

import csv
import numpy as np
import statistics
from binance.client import Client
from utils import get_top_symbols  # Import get_top_symbols for consistency
from config import API_KEY, API_SECRET

# Initialize Binance client
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'
def collect_data(symbol, interval='3m', limit=100):
    """
    Collect short-term data for scalping strategy with 3-minute intervals.
    Saves data with indicators that capture rapid price movements.
    """
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        if len(klines) < 15:
            print(f"Not enough data for {symbol}. Skipping.")
            return
        
        with open(f"data/{symbol}_scalping_data.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Open", "High", "Low", "Close", "Volume", "MA3", "MA5", "MA15", "RSI", "Volatility", "Trend"])

            for i in range(15, len(klines)):
                close_prices = [float(k[4]) for k in klines[i-15:i]]

                # Calculate moving averages
                ma3 = np.nanmean(close_prices[-3:]) if len(close_prices[-3:]) > 0 else 0
                ma5 = np.nanmean(close_prices[-5:]) if len(close_prices[-5:]) > 0 else 0
                ma15 = np.nanmean(close_prices) if len(close_prices) > 0 else 0

                # Calculate gains and losses for RSI
                diff = np.diff(close_prices)
                gains = np.nanmean([d for d in diff if d > 0]) if len([d for d in diff if d > 0]) > 0 else 0
                losses = np.nanmean([-d for d in diff if d < 0]) if len([d for d in diff if d < 0]) > 0 else 0
                rsi = 100 - (100 / (1 + gains / losses)) if losses != 0 else 100

                # Calculate volatility
                volatility = statistics.stdev(close_prices) if len(close_prices) > 1 else 0

                # Determine trend (1 if price increased, 0 if decreased)
                trend = 1 if close_prices[-1] > close_prices[0] else 0

                # Write row to CSV
                row = klines[i][:5] + [ma3, ma5, ma15, rsi, volatility, trend]
                writer.writerow(row)

    except Exception as e:
        print(f"Error collecting data for {symbol}: {e}")

if __name__ == "__main__":
    symbols = get_top_symbols(limit=50, profit_target=0.005)  # Automatically fetch top symbols
    for symbol in symbols:
        collect_data(symbol)
        print(f"Data collection complete for {symbol}.")

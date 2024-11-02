# utils.py

from binance.client import Client
import statistics
from config import API_KEY,API_SECRET

# Initialize Binance client
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'

def get_top_symbols(limit=15, profit_target=0.003):
    """
    Fetch top symbols based on volume, volatility, and stability for scalping.
    """
    tickers = client.get_ticker()
    sorted_tickers = sorted(tickers, key=lambda x: float(x['quoteVolume']), reverse=True)
    top_symbols = []
    for ticker in sorted_tickers:
        if ticker['symbol'].endswith("USDT"):
            try:
                klines = client.get_klines(symbol=ticker['symbol'], interval=Client.KLINE_INTERVAL_15MINUTE, limit=30)
                closing_prices = [float(kline[4]) for kline in klines]
                stddev = statistics.stdev(closing_prices)
                
                avg_price = sum(closing_prices) / len(closing_prices)
                volatility_ratio = stddev / avg_price

                if stddev < 0.03 and volatility_ratio >= profit_target:
                    top_symbols.append(ticker['symbol'])
                    print(f"تم اختيار العملة {ticker['symbol']} بنسبة تذبذب {volatility_ratio:.4f}")
                
                if len(top_symbols) >= limit:
                    break
            except Exception as e:
                print(f"خطأ في جلب بيانات {ticker['symbol']}: {e}")
    return top_symbols

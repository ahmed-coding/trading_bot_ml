# bot_with_ml.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from binance.client import Client
import statistics
import time
from datetime import datetime
from utils import get_top_symbols  # Import get_top_symbols for consistency
import threading  # For concurrent trade operations
from config import API_KEY,API_SECRET
import math


# Initialize Binance client
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'

# Load ML models
trend_model = load_model("models/lstm_trend_predictor_scalping.keras")
volatility_model = joblib.load("models/volatility_predictor_scalping.pkl")
scaler = joblib.load("models/scaler.pkl")

# Parameters for scalping
base_profit_target = 0.0032  # Base profit target
base_stop_loss = 0.001      # Base stop loss
investment = 10             # Fixed investment per trade
timeout = 300               # Transaction timeout in seconds (5 minutes)
commission_rate = 0.001  # Commission rate for Binance

# Initialize balance
balance = 101  # Starting balance (adjust based on your actual balance)

# Dictionary to store active trades
active_trades = {}
excluded_symbols = set()  # Track symbols causing frequent errors
symbols_to_trade = []  # List to store symbols selected for trading


def adjust_balance(amount, action="buy"):
    """
    Adjust the balance after a buy or sell action, accounting for commission.
    """
    global balance
    commission = amount * commission_rate
    if action == "buy":
        balance -= (amount + commission)
    elif action == "sell":
        balance += amount - commission
    print(f"تم تحديث الرصيد بعد {action} - الرصيد المتبقي: {balance}")

def check_balance(investment):
    """
    Check if there is enough balance to open a trade.
    """
    return balance >= investment

# Feature preparation for ML models
def prepare_features(symbol):
    """
    Prepare features for the ML models using recent price data.
    """
    klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=10)
    close_prices = [float(k[4]) for k in klines]

    # Check if there are enough close prices to calculate features
    if len(close_prices) < 5:
        print(f"{symbol} - بيانات غير كافية لحساب الميزات.")
        return None, None  # Return None to indicate insufficient data
    
    ma3 = np.mean(close_prices[-3:])
    ma5 = np.mean(close_prices[-5:])
    
    # Improved RSI calculation with epsilon to avoid divide-by-zero errors
    epsilon = 1e-10  # Small constant to prevent divide-by-zero
    gains = np.mean([diff for diff in np.diff(close_prices) if diff > 0]) + epsilon
    losses = np.mean([-diff for diff in np.diff(close_prices) if diff < 0]) + epsilon
    rsi = 100 - (100 / (1 + gains / losses))
    
    volatility = statistics.stdev(close_prices[-5:])
    
    # Convert features to DataFrame to preserve column names
    features = pd.DataFrame([[ma3, ma5, rsi, volatility]], columns=['MA3', 'MA5', 'RSI', 'Volatility'])
    
    # Scale features
    scaled_features = scaler.transform(features)
    return scaled_features, close_prices


# In predict_trend
def predict_trend(symbol):
    features, _ = prepare_features(symbol)
    if features is None:
        print(f"{symbol} - فشل في توقع الاتجاه بسبب البيانات الناقصة.")
        return None  # Skip if data is insufficient
    trend_prediction = trend_model.predict(features)
    return trend_prediction[0][0] > 0.5


# In predict_volatility
def predict_volatility(symbol):
    features, _ = prepare_features(symbol)
    if features is None:
        print(f"{symbol} - فشل في توقع التذبذب بسبب البيانات الناقصة.")
        return None  # Skip if data is insufficient
    predicted_volatility = volatility_model.predict(features)[0]
    return predicted_volatility



def get_top_symbols(limit=10, profit_target=0.003):
    """
    Fetch top symbols based on volume, volatility, and stability for scalping.
    """
    tickers = client.get_ticker()
    sorted_tickers = sorted(tickers, key=lambda x: float(x['quoteVolume']), reverse=True)
    top_symbols = []
    for ticker in sorted_tickers:
        if ticker['symbol'].endswith("USDT") and ticker['symbol'] not in excluded_symbols:
            try:
                klines = client.get_klines(symbol=ticker['symbol'], interval=Client.KLINE_INTERVAL_1MINUTE, limit=30)
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
                excluded_symbols.add(ticker['symbol'])
    return top_symbols

def get_lot_size(symbol):
    exchange_info = client.get_symbol_info(symbol)
    for filter in exchange_info['filters']:
        if filter['filterType'] == 'LOT_SIZE':
            step_size = float(filter['stepSize'])
            return step_size
    return None

def adjust_quantity(symbol, quantity):
    """
    Adjusts the quantity to match Binance's required precision for the symbol.
    """
    step_size = get_lot_size(symbol)
    if step_size:
        precision = int(round(-math.log(step_size, 10), 0))
        quantity = round(quantity, precision)
        # Ensure quantity is a multiple of step size
        quantity = math.floor(quantity / step_size) * step_size
    return quantity



def get_min_notional(symbol):
    """
    Retrieve the minimum notional value for a symbol from Binance filters.
    """
    filters = client.get_symbol_info(symbol)['filters']
    for filter in filters:
        if filter['filterType'] == 'MIN_NOTIONAL':
            return float(filter['minNotional'])
    return None  # Return None if minNotional is not found




def open_trade_with_ml(symbol):
    """
    Open a trade with a dynamic profit target and stop loss based on ML predictions.
    """
    global balance
    current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
    min_notional = get_min_notional(symbol)
    
    # Check if min_notional was found; skip trade if not
    # if min_notional is None:
    #     print(f"{symbol} - لا يوجد حد أدنى للصفقة (minNotional) لهذا الرمز.")
    #     return
    
    # Calculate investment amount and check if it meets notional requirement
    notional_value = investment / current_price
    if balance < investment:
        print(f"{symbol} - الرصيد غير كافٍ أو قيمة الصفقة أقل من الحد الأدنى المسموح.")
        return
    
    # Predict trend direction and skip trade if trend is down
    trend_up = predict_trend(symbol)
    if trend_up is None or not trend_up:
        print(f"{symbol} - الاتجاه هابط أو البيانات غير كافية، لن يتم فتح الصفقة.")
        return

    # Predict volatility and set profit target and stop-loss
    predicted_volatility = predict_volatility(symbol)
    if predicted_volatility is None:
        print(f"{symbol} - فشل في توقع التذبذب، لن يتم فتح الصفقة.")
        return

    profit_target = base_profit_target + (predicted_volatility * 0.5)
    stop_loss = base_stop_loss - (predicted_volatility * 0.3)
    target_price = current_price * (1 + profit_target)
    stop_price = current_price * (1 - stop_loss)
    
    # Calculate quantity and adjust for LOT_SIZE
    quantity = adjust_quantity(symbol, investment / current_price)
    if quantity == 0:
        print(f"{symbol} - الكمية بعد الضبط أقل من الحد الأدنى المطلوب.")
        return
    
    # Open the trade with adjusted quantity
    try:
        open_trade(symbol, quantity, target_price, stop_price)
    except Exception as e:
        print(f"خطأ في فتح الصفقة لـ {symbol}: {e}")
        
        
def open_trade(symbol, quantity, target_price, stop_price):
    """
    Executes a buy order with defined target and stop prices and starts tracking the trade.
    """
    global balance
    try:
        # Execute market buy order
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
        
        # Adjust balance for investment and commission
        investment_amount = quantity * float(client.get_symbol_ticker(symbol=symbol)['price'])
        commission = investment_amount * commission_rate
        balance -= (investment_amount + commission)
        
        # Store trade details
        active_trades[symbol] = {
            'quantity': quantity,
            'initial_price': investment_amount / quantity,
            'target_price': target_price,
            'stop_price': stop_price,
            'start_time': time.time(),
            'timeout': timeout
        }
        
        print(f"{datetime.now()} - تم فتح صفقة شراء لـ {symbol} بكمية {quantity}, بهدف {target_price} وإيقاف خسارة عند {stop_price}")
    except Exception as e:
        print(f"خطأ في فتح الصفقة لـ {symbol}: {e}")


def sell_trade(symbol, trade_quantity):
    """
    Execute a sell trade for the specified quantity, adjusting to meet LOT_SIZE requirements.
    """
    try:
        # Get available quantity and ensure it meets LOT_SIZE
        step_size = get_lot_size(symbol)
        adjusted_quantity = math.floor(trade_quantity / step_size) * step_size
        
        if adjusted_quantity < step_size:
            print(f"{symbol} - الكمية بعد التقريب ({adjusted_quantity}) لا تزال أقل من الحد الأدنى المطلوب لـ LOT_SIZE ({step_size}).")
            return 0

        # Execute market sell order
        client.order_market_sell(symbol=symbol, quantity=adjusted_quantity)
        print(f"تم تنفيذ عملية البيع لـ {symbol} بكمية {adjusted_quantity}")
        return adjusted_quantity
    except Exception as e:
        print(f"خطأ في بيع {symbol}: {e}")
        return 0



# Monitor active trades to execute sell orders if conditions are met
def check_trade_conditions():
    """
    Monitors active trades, closing them if they meet profit target, stop-loss, or timeout conditions.
    """
    for symbol, trade in list(active_trades.items()):
        try:
            current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
            result = None
            
            # Check if any close condition is met
            if current_price >= trade['target_price']:
                result = "ربح"
            elif current_price <= trade['stop_price']:
                result = "خسارة"
            elif time.time() - trade['start_time'] >= trade['timeout']:
                result = "انتهاء المهلة"
            
            # Execute sell and update balance
            if result:
                sell_quantity = sell_trade(symbol, trade['quantity'])
                if sell_quantity > 0:
                    sale_amount = sell_quantity * current_price
                    adjust_balance(sale_amount, action="sell")
                    print(f"{datetime.now()} - تم {result} الصفقة لـ {symbol} بسعر {current_price}")
                del active_trades[symbol]
        except Exception as e:
            print(f"خطأ في التحقق من الشروط للصفقة {symbol}: {e}")


def execute_sell(symbol, quantity, price, result):
    """
    Execute a market sell order and remove the trade from active_trades.
    """
    try:
        # Execute the sell order
        client.order_market_sell(symbol=symbol, quantity=quantity)
        
        # Adjust balance after selling
        sale_amount = quantity * price
        adjust_balance(sale_amount, action="sell")
        
        # Log the trade result
        print(f"{datetime.now()} - تم {result} الصفقة لـ {symbol} بسعر {price}")
        
        # Remove the trade from active trades
        del active_trades[symbol]
    except Exception as e:
        print(f"خطأ في تنفيذ البيع للصفقة {symbol}: {e}")

# Periodically check and update active trades
# def monitor_trades():
#     """
#     Continuously checks for conditions to close active trades.
#     """
#     while True:
#         check_trade_conditions()
#         # time.sleep()

# # Run the main trading loop to open and monitor trades

# Define symbols_to_trade as a global list for storing top symbols separately

def update_symbols_periodically(interval=900):
    """
    Periodically updates the list of top symbols to trade.
    """
    global symbols_to_trade
    while True:
        symbols_to_trade = get_top_symbols(limit=10, profit_target=0.003)
        print(f"{datetime.now()} - تم تحديث قائمة العملات للتداول: {symbols_to_trade}")
        time.sleep(interval)

def update_prices():
    """
    Continuously update prices and open new trades based on ML predictions.
    """
    global active_trades
    for symbol in symbols_to_trade:
        if symbol in excluded_symbols or symbol in active_trades:
            continue
        try:
            current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
            print(f"تم تحديث السعر لعملة {symbol}: {current_price}")
            
            # Open a trade if it's not already active for the symbol
            open_trade_with_ml(symbol)
                
        except Exception as e:
            print(f"خطأ في تحديث السعر لـ {symbol}: {e}")
            excluded_symbols.add(symbol)  # Exclude symbols causing frequent errors

def monitor_trades():
    """
    Continuously check for conditions to close active trades.
    """
    while True:
        check_trade_conditions()
        # time.sleep(1)  # Optional: add a short delay for efficient processing

# Main function to start the bot
def start_trading():
    """
    Starts the trading bot by initializing threads for symbol updates, price monitoring, and trade monitoring.
    """
    # Periodic symbol updates
    symbol_update_thread = threading.Thread(target=update_symbols_periodically, args=(900,))
    symbol_update_thread.start()

    # Price updates and trade openings
    price_thread = threading.Thread(target=update_prices)
    price_thread.start()

    # Trade monitoring
    trade_thread = threading.Thread(target=monitor_trades)
    trade_thread.start()

    print(f"تم بدء تشغيل البوت في {datetime.now()}")


if __name__ == "__main__":
    start_trading()

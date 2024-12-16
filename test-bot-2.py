import json
from binance.client import Client
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import concurrent.futures  # Import concurrent.futures
# from utils import get_top_symbols  # Import get_top_symbols for consistency
from datetime import datetime
import math
import time
import csv
import os
import statistics
from binance.exceptions import BinanceAPIException
import threading
from threading import Lock

import requests
import warnings

from config import API_KEY, API_SECRET, Settings
from train_models_test_2 import TrendModelManager, VolatilityModelManager, DataLoader, TradeLogModelManager, RetrainManager

warnings.filterwarnings("ignore", category=RuntimeWarning)

session = requests.Session()
session.headers.update({'timeout': '90'})

# إعداد مفاتيح API الخاصة بك
client = Client(API_KEY, API_SECRET)
# client.API_URL = 'https://testnet.binance.vision/api'  # استخدام منصة بايننس التجريبية

# إدارة المحفظة
balance = 80  # الرصيد المبدئي للبوت
investment = 8  # حجم كل صفقة (تم تقليله لتقليل المخاطر)
base_profit_target = 0.004  # نسبة الربح
base_stop_loss = 0.001  # نسبة الخسارة
max_open_trades = 9  # الحد الأقصى للصفقات المفتوحة في نفس الوقت (تم تقليله لتقليل المخاطر)
timeout = 5  # وقت انتهاء وقت الصفقة
commission_rate = 0.001  # نسبة العمولة للمنصة
excluded_symbols = set()  # قائمة العملات المستثناة بسبب أخطاء متكررة
symbols_to_trade = []
current_prices = {}
active_trades = {}
lose_symbols = set()
bot_settings = Settings()

# ملف CSV لتسجيل التداولات
csv_file = 'trades_log.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['الرمز', 'الكمية', 'السعر الابتدائي', 'سعر الهدف', 'سعر الإيقاف', 'الوقت', 'النتيجة', 'الرصيد المتبقي'])

# --------------------- ML ------------------
print("تحميل النماذج...")
trend_model_manager = TrendModelManager("models/lstm_trend_predictor_scalping.keras", "models/scaler.pkl")
volatility_model_manager = VolatilityModelManager("models/volatility_predictor_scalping.pkl", "models/scaler.pkl")
trade_log_model_manager = TradeLogModelManager("models/trade_log_scalping.keras", "models/scaler_trade_log.pkl")
retrain_manager = RetrainManager(trend_model_manager, volatility_model_manager, trade_log_model_manager, DataLoader(data_directory="data"))
model_lock = Lock()

# Prepare features for ML models
def prepare_features(symbol):
    try:
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_3MINUTE, limit=30)
        close_prices = [float(k[4]) for k in klines]
        if len(close_prices) < 15:
            print(f"{symbol} - بيانات غير كافية لحساب الميزات.")
            return None, None
        ma3 = np.mean(close_prices[-3:])
        ma5 = np.mean(close_prices[-5:])
        ma15 = np.mean(close_prices)
        gains = np.mean([diff for diff in np.diff(close_prices) if diff > 0]) if np.diff(close_prices).any() else 0
        losses = np.mean([-diff for diff in np.diff(close_prices) if diff < 0]) if np.diff(close_prices).any() else 0
        rsi = 100 - (100 / (1 + gains / losses)) if losses != 0 else 100
        volatility = statistics.stdev(close_prices)
        features = pd.DataFrame([[ma3, ma5, ma15, rsi, volatility]], columns=['MA3', 'MA5', 'MA15', 'RSI', 'Volatility'])
        scaled_features = trend_model_manager.scaler.transform(features)
        return scaled_features, close_prices
    except Exception as e:
        print(f"{symbol} - خطأ في التحضير للميزات: {e}")
        return None, None

# Prediction helper functions
def predict_trend(features):
    try:
        with model_lock:
            trend_prediction = trend_model_manager.model.predict(features, verbose=0)
        return trend_prediction[0][0] > 0.50
    except Exception as e:
        print(f"خطأ في توقع الاتجاه: {e}")
        return False

def predict_volatility(features):
    try:
        with model_lock:
            features = np.array(features).reshape(1, -1)  # Reshape to match expected input
            predicted_volatility = volatility_model_manager.model.predict(features)[0]
        return predicted_volatility
    except Exception as e:
        print(f"خطأ في توقع التذبذب: {e}")
        return None

# فتح صفقة بناءً على الهدف الديناميكي
def open_trade_with_dynamic_target(symbol):
    global balance

    if bot_settings.trading_status() == "0":
        print("the trading is off, can't open more trades")
        return

    if len(active_trades) >= max_open_trades:
        return

    features, _ = prepare_features(symbol)
    if features is None:
        return

    trend_is_up = predict_trend(features)
    if not trend_is_up:
        return

    predicted_volatility = predict_volatility(features)
    if predicted_volatility is None:
        return

    if not check_bnb_balance():
        print(f"{datetime.now()} - الرصيد غير كافٍ من BNB لتغطية الرسوم. يرجى إيداع BNB.")
        return

    current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
    profit_target = base_profit_target + predicted_volatility * 0.3
    stop_loss = base_stop_loss + predicted_volatility * 0.02
    target_price = current_price * (1 + profit_target)
    stop_price = current_price * (1 - stop_loss)
    quantity = adjust_quantity(symbol, investment / current_price)
    if balance < investment:
        print(f"الرصيد غير كافٍ لفتح صفقة جديدة لـ {symbol}.")
        return
    try:
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
        adjust_balance(investment, action="buy")
        active_trades[symbol] = {
            'quantity': quantity,
            'initial_price': current_price,
            'target_price': target_price,
            'stop_price': stop_price,
            'start_time': time.time(),
            'timeout': timeout * 60
        }
        print(f"تم فتح صفقة شراء لـ {symbol} بسعر {current_price} بهدف {target_price} وإيقاف خسارة عند {stop_price}")
    except BinanceAPIException as e:
        print(f"خطأ في فتح الصفقة لـ {symbol}: {e}")
        excluded_symbols.add(symbol)

# تحديث قائمة العملات بشكل متسلسل
def update_symbols():
    global symbols_to_trade
    symbols_to_trade = get_top_symbols(50)
    print(f"{datetime.now()} - تم تحديث قائمة العملات للتداول: {symbols_to_trade}")
    retrain_manager.retrain_models_if_needed()

# تحديث الأسعار وفتح الصفقات بشكل متسلسل
def update_prices_and_open_trades():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(update_price_and_open_trade, symbol) for symbol in symbols_to_trade if symbol not in excluded_symbols]
        concurrent.futures.wait(futures)

def update_price_and_open_trade(symbol):
    try:
        current_prices[symbol] = float(client.get_symbol_ticker(symbol=symbol)['price'])
        print(f"تم تحديث السعر لعملة {symbol}: {current_prices[symbol]}")

        if symbol not in active_trades:
            open_trade_with_dynamic_target(symbol)
    except BinanceAPIException as e:
        print(f"خطأ في تحديث السعر لـ {symbol}: {e}")
        if 'NOTIONAL' in str(e) or 'Invalid symbol' in str(e):
            excluded_symbols.add(symbol)  # استبعاد العملات التي تسبب أخطاء متكررة

# المراقبة وإغلاق الصفقات
def monitor_trades():
    global balance
    for symbol, trade in list(active_trades.items()):
        try:
            current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
            result = None

            if current_price >= trade['target_price']:
                result = "ربح"
            elif current_price <= trade['stop_price']:
                result = "خسارة"
                lose_symbols.add(symbol)
            elif time.time() - trade['start_time'] >= trade['timeout']:
                result = "انتهاء المهلة"

            if result:
                sell_trade(symbol, trade['quantity'], result)
                adjust_balance(current_price * trade['quantity'], action='sell')
                print(f"{datetime.now()} - تم {result} الصفقة لـ {symbol} عند السعر {current_price}")
                with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([symbol, trade['quantity'], trade['initial_price'], trade['target_price'], trade['stop_price'], datetime.now(), result, balance])
                del active_trades[symbol]
        except BinanceAPIException as e:
            print(f"خطأ في التحقق من الشروط للصفقة {symbol}: {e}")
            if 'NOTIONAL' in str(e) or 'Invalid symbol' in str(e):
                excluded_symbols.add(symbol)

# ضبط الرصيد بناءً على العملية (شراء أو بيع) وخصم العمولة
def adjust_balance(amount, action="buy"):
    global balance, commission_rate
    commission = amount * commission_rate
    if action == "buy":
        balance -= (amount + commission)  # خصم المبلغ + العمولة
    elif action == "sell":
        balance += amount - commission  # إضافة المبلغ بعد خصم العمولة
    print(f"تم تحديث الرصيد بعد {action} - الرصيد المتبقي: {balance}")
    return balance

# دالة لضبط الكمية بناءً على دقة السوق
def adjust_quantity(symbol, quantity):
    step_size = get_lot_size(symbol)
    if step_size is None:
        return quantity
    precision = int(round(-math.log(step_size, 10), 0))
    return round(quantity, precision)

# دالة للحصول على حجم اللوت للرمز المحدد
def get_lot_size(symbol):
    exchange_info = client.get_symbol_info(symbol)
    for filter in exchange_info['filters']:
        if filter['filterType'] == 'LOT_SIZE':
            step_size = float(filter['stepSize'])
            return step_size
    return None

# التحقق من رصيد BNB لتغطية الرسوم
def check_bnb_balance(min_bnb_balance=0.001):
    account_info = client.get_asset_balance(asset='BNB')
    if account_info:
        bnb_balance = float(account_info['free'])
        return bnb_balance >= min_bnb_balance
    return False

# دالة لبيع الصفقة بناءً على النتيجة
def sell_trade(symbol, quantity, result):
    try:
        client.order_market_sell(symbol=symbol, quantity=quantity)
        print(f"{datetime.now()} - تم {result} الصفقة لـ {symbol}")
    except BinanceAPIException as e:
        print(f"خطأ في بيع {symbol}: {e}")

# دالة للحصول على أفضل العملات
def get_top_symbols(limit=50, profit_target=0.007):
    tickers = client.get_ticker()
    sorted_tickers = sorted(tickers, key=lambda x: float(x['quoteVolume']), reverse=True)
    top_symbols = []
    lose_symbols = set()
    for ticker in sorted_tickers:
        if ticker['symbol'].endswith("USDC") and ticker['symbol'] not in excluded_symbols:
            try:
                klines = client.get_klines(symbol=ticker['symbol'], interval=Client.KLINE_INTERVAL_3MINUTE, limit=30)
                closing_prices = [float(kline[4]) for kline in klines]
                stddev = statistics.stdev(closing_prices)
                avg_price = sum(closing_prices) / len(closing_prices)
                volatility_ratio = stddev / avg_price

                if stddev < 0.03:
                    top_symbols.append(ticker['symbol'])
                    print(f"تم اختيار العملة {ticker['symbol']} بنسبة تذبذب {volatility_ratio:.4f}")
                if len(top_symbols) >= limit:
                    break
            except BinanceAPIException as e:
                print(f"خطأ في جلب بيانات {ticker['symbol']}: {e}")
                excluded_symbols.add(ticker['symbol'])
    return top_symbols

# بدء تشغيل البوت
def run_bot():
    last_symbol_update_time = time.time() - 900  # فرض تحديث أولي فور بدء التشغيل
    while True:
        if bot_settings.bot_status() == '0':
            print("Bot is turn off")
            return
        current_time = time.time()
        if current_time - last_symbol_update_time >= 900:
            update_symbols()
            last_symbol_update_time = current_time
        update_prices_and_open_trades()
        monitor_trades()
        retrain_manager.retrain_models_if_needed()  # استدعاء دالة إعادة التدريب بعد كل دورة
        time.sleep(1)

if __name__ == "__main__":
    run_bot()

# test-loop-3-test.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime
import math
import time
import csv
import os
import statistics
from binance.exceptions import BinanceAPIException
import requests
import warnings
from threading import Lock
from binance.client import Client
import concurrent.futures
import threading
from config import API_KEY, API_SECRET,Settings
from train_models import train_trend_model, train_volatility_model, load_data, retrain_models_if_needed
from sklearn.impute import SimpleImputer


warnings.filterwarnings("ignore", category=RuntimeWarning)

session = requests.Session()
session.headers.update({'timeout': '90'})

# إعداد مفاتيح API الخاصة بك
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'  # استخدام منصة بايننس التجريبية

# إدارة المحفظة
balance = 100  # الرصيد المبدئي للبوت
investment = 10  # حجم كل صفقة (تم تقليله لتقليل المخاطر)
base_profit_target = 0.005 # زيادة نسبة الربح لتحسين الاستراتيجية
base_stop_loss = 0.001  # تعديل نسبة الخسارة لتكون أكثر صرامة
max_open_trades = 9  # الحد الأقصى للصفقات المفتوحة في نفس الوقت (تم تقليله لتقليل المخاطر)
timeout = 5  # وقت انتهاء وقت الصفقة (بالدقائق)
commission_rate = 0.001  # نسبة العمولة للمنصة
excluded_symbols = set()  # قائمة العملات المستثناة بسبب أخطاء متكررة
symbols_to_trade = []
current_prices = {}
active_trades = {}
lose_symbols = set()
bot_settings=Settings()

# ملف CSV لتسجيل التداولات
csv_file = 'trades_log_test.csv'  # حفظ النتائج في ملف آخر للتجربة
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['الرمز', 'الكمية', 'السعر الابتدائي', 'سعر الهدف', 'سعر الإيقاف', 'الوقت', 'النتيجة', 'الرصيد المتبقي'])

# --------------------- ML ------------------
# تحميل النماذج فقط مرة واحدة مع استخدام القفل
print("تحميل النماذج...")
model_lock = Lock()

def load_models():
    global trend_model, volatility_model, scaler
    with model_lock:
        trend_model = load_model("models/lstm_trend_predictor_scalping.keras")
        volatility_model = joblib.load("models/volatility_predictor_scalping.pkl")
        scaler = joblib.load("models/scaler.pkl")
    print("تم تحميل النماذج بنجاح.")

load_models()

# # Prepare features for ML models
# def prepare_features(symbol):
#     try:
#         klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_3MINUTE, limit=15)
#         close_prices = [float(k[4]) for k in klines]
#         if len(close_prices) < 15:
#             print(f"{symbol} - بيانات غير كافية لحساب الميزات.")
#             return None, None

#         ma3 = np.mean(close_prices[-3:])
#         ma5 = np.mean(close_prices[-5:])
#         ma15 = np.mean(close_prices)
#         gains = np.mean([diff for diff in np.diff(close_prices) if diff > 0]) if len([diff for diff in np.diff(close_prices) if diff > 0]) > 0 else 0
#         losses = np.mean([-diff for diff in np.diff(close_prices) if diff < 0]) if len([diff for diff in np.diff(close_prices) if diff < 0]) > 0 else 0
#         rsi = 100 - (100 / (1 + gains / losses)) if losses != 0 else 100
#         volatility = statistics.stdev(close_prices)

#         features = pd.DataFrame([[ma3, ma5, ma15, rsi, volatility]], columns=['MA3', 'MA5', 'MA15', 'RSI', 'Volatility'])
#         if features.isnull().values.any():
#             print(f"{symbol} - بيانات تحتوي على قيم مفقودة. سيتم تجاهل هذا الرمز.")
#             return None, None

#         imputer = SimpleImputer(strategy='mean')
#         features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

#         # تحجيم الميزات باستخدام scaler المحمل من النموذج
#         features_scaled = scaler.transform(features)

#         return features_scaled, close_prices
#     except Exception as e:
#         print(f"{symbol} - خطأ في التحضير للميزات: {e}")
#         return None, None

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
        scaled_features = scaler.transform(features)
        return scaled_features, close_prices
    except Exception as e:
        print(f"{symbol} - خطأ في التحضير للميزات: {e}")
        return None, None


# Prediction helper functions
def predict_trend(features):
    try:
        with model_lock:
            trend_prediction = trend_model.predict(features, verbose=0)
        return trend_prediction[0][0] > 0.60  # زيادة العتبة للتوقع الصاعد لتجنب الصفقات الخاسرة
    except Exception as e:
        print(f"خطأ في توقع الاتجاه: {e}")
        return False

def predict_volatility(features):
    try:
        with model_lock:
            features = np.array(features).reshape(1, -1)  # Reshape to match expected input
            predicted_volatility = volatility_model.predict(features)[0]
        return predicted_volatility
    except Exception as e:
        print(f"خطأ في توقع التذبذب: {e}")
        return None
# إعادة تحميل النماذج عند الحاجة
def reload_models_if_updated():
    try:
        current_trend_model_time = os.path.getmtime("models/lstm_trend_predictor_scalping.keras")
        current_volatility_model_time = os.path.getmtime("models/volatility_predictor_scalping.pkl")
        current_scaler_time = os.path.getmtime("models/scaler.pkl")

        global trend_model_last_loaded, volatility_model_last_loaded, scaler_last_loaded
        if (trend_model_last_loaded is None or trend_model_last_loaded < current_trend_model_time or
            volatility_model_last_loaded is None or volatility_model_last_loaded < current_volatility_model_time or
            scaler_last_loaded is None or scaler_last_loaded < current_scaler_time):
            load_models()
            trend_model_last_loaded = current_trend_model_time
            volatility_model_last_loaded = current_volatility_model_time
            scaler_last_loaded = current_scaler_time
    except Exception as e:
        print(f"خطأ في التحقق من تحديث النماذج: {e}")

trend_model_last_loaded = None
volatility_model_last_loaded = None
scaler_last_loaded = None
reload_models_if_updated()

# تدريب النماذج باستخدام البيانات الجديدة
def train_models_with_updated_data():
    print("تدريب النماذج باستخدام البيانات المحدثة...")
    features, trend_target, volatility_target, scaler = load_data()

    # Split for training and testing
    X_train, X_test, y_train_trend, y_test_trend = train_test_split(features, trend_target, test_size=0.2, random_state=42)
    _, _, y_train_volatility, y_test_volatility = train_test_split(features, volatility_target, test_size=0.2, random_state=42)

    # Train the trend model (LSTM)
    trend_model = train_trend_model(X_train, y_train_trend)

    # Train the volatility model (Gradient Boosting)
    volatility_model = train_volatility_model(X_train, y_train_volatility)

    # Save models and scaler
    trend_model.save("models/lstm_trend_predictor_scalping.keras")  # Save as .keras
    joblib.dump(volatility_model, "models/volatility_predictor_scalping.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Model training and saving complete.")

# --------------------------------

# فتح صفقة بناءً على الهدف الديناميكي
def open_trade_with_dynamic_target(symbol):
    
    if bot_settings.trading_status() =="0":
        print("the trading is of can't open more trad")
        return
    
    global balance
    if len(active_trades) >= max_open_trades:
        return

    features, _ = prepare_features(symbol)
    if features is None:
        return

    # تحديد الاتجاه بناءً على التنبؤ
    trend_is_up = predict_trend(features)
    if not trend_is_up:
        return

    predicted_volatility = predict_volatility(features)
    if predicted_volatility is None:
        return
    if not check_bnb_balance():
        print(f"{datetime.now()} - الرصيد غير كافٍ من BNB لتغطية الرسوم. يرجى إيداع BNB.")
        return
    # if symbol in lose_symbols:
    #     return
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
    # Train models with updated data
    train_models_with_updated_data()

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
    lose_symbols=set()
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
            print("Bot is turn of")
            return
        current_time = time.time()
        if current_time - last_symbol_update_time >= 900:
            update_symbols()
            last_symbol_update_time = current_time
        update_prices_and_open_trades()
        monitor_trades()
        retrain_models_if_needed()  # استدعاء دالة إعادة التدريب بعد كل دورة
        time.sleep(1)


if __name__ == "__main__":
    run_bot()

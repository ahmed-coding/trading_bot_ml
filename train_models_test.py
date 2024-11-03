import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import joblib
import time
from threading import Lock
import warnings
import requests
import csv

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Directory containing scalping data
data_directory = "data"  # Folder where data files are stored

# Training interval (e.g., retrain every 1 hour)
RETRAIN_INTERVAL = 900  # 1 hour in seconds
last_training_time = 0

# Model lock for thread safety
model_lock = Lock()

# Load models if they exist
print("Loading models...")
try:
    with model_lock:
        trend_model = load_model("models/lstm_trend_predictor_scalping.keras")
        volatility_model = joblib.load("models/volatility_predictor_scalping.pkl")
        scaler = joblib.load("models/scaler.pkl")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Models could not be loaded: {e}")
    trend_model, volatility_model, scaler = None, None, None


def load_data():
    all_data = []
    for file_name in os.listdir(data_directory):
        if file_name.endswith("_scalping_data.csv"):
            file_path = os.path.join(data_directory, file_name)
            data = pd.read_csv(file_path)
            if 'Trend' not in data.columns:
                data['Trend'] = (data['MA5'] > data['MA15']).astype(int)  # Example trend calculation
            all_data.append(data)
            print(f"Loaded data from {file_name}")

    combined_data = pd.concat(all_data, ignore_index=True)

    # Drop rows where any feature values are missing
    combined_data.dropna(subset=["MA3", "MA5", "MA15", "RSI", "Volatility"], inplace=True)

    features = combined_data[["MA3", "MA5", "MA15", "RSI", "Volatility"]]
    trend_target = combined_data["Trend"]
    volatility_target = combined_data["Volatility"]

    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)

    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

    # Ensure the length of targets matches the features
    min_length = min(len(features_scaled_df), len(trend_target), len(volatility_target))
    features_scaled_df = features_scaled_df.iloc[:min_length]
    trend_target = trend_target.iloc[:min_length]
    volatility_target = volatility_target.iloc[:min_length]

    return features_scaled_df, trend_target, volatility_target, scaler


def load_online_data():
    """
    Load and preprocess data from online sources to enhance model training.
    """
    try:
        response = requests.get("https://api.binance.com/api/v3/ticker/24hr")  # Example API endpoint
        response.raise_for_status()
        market_data = response.json()

        # Filter the necessary data (e.g., close price, volume, etc.)
        records = []
        for data in market_data:
            if data['symbol'].endswith('USDT'):  # Assuming we want USDT pairs
                close_price = float(data['lastPrice'])
                high_price = float(data['highPrice'])
                low_price = float(data['lowPrice'])
                volume = float(data['volume'])

                records.append([close_price, high_price, low_price, volume])

        # Convert to DataFrame
        df = pd.DataFrame(records, columns=['Close', 'High', 'Low', 'Volume'])

        # Feature engineering
        df['MA3'] = df['Close'].rolling(window=3).mean()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA15'] = df['Close'].rolling(window=15).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['Volatility'] = df['Close'].rolling(window=15).std()

        # Drop rows with missing values (due to rolling calculations)
        df.dropna(inplace=True)

        # Prepare features and targets
        features = df[['MA3', 'MA5', 'MA15', 'RSI', 'Volatility']]
        trend_target = (df['Close'].shift(-1) > df['Close']).astype(int)  # Example trend calculation
        volatility_target = df['Volatility']

        return features, trend_target, volatility_target
    except Exception as e:
        print(f"Error loading online data: {e}")
        return None, None, None


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def load_trade_log_data():
    trade_log_file = "trades_log_test.csv"
    if not os.path.exists(trade_log_file):
        print("Trade log file not found.")
        return None, None
    try:
        trade_log = pd.read_csv(trade_log_file)
        features = trade_log[["السعر الابتدائي", "سعر الهدف", "سعر الإيقاف"]]
        target = trade_log["النتيجة"].apply(lambda x: 1 if x == "ربح" else 0)

        imputer = SimpleImputer(strategy='mean')
        features_imputed = imputer.fit_transform(features)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_imputed)

        features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

        return features_scaled_df, target
    except Exception as e:
        print(f"Error loading trade log data: {e}")
        return None, None


def train_trend_model(X_train, y_train):
    """
    Train LSTM model for predicting short-term price trend.
    """
    trend_model = Sequential()
    trend_model.add(Input(shape=(X_train.shape[1], 1)))
    trend_model.add(LSTM(128, return_sequences=True))
    trend_model.add(Dropout(0.3))
    trend_model.add(LSTM(128))
    trend_model.add(Dropout(0.3))
    trend_model.add(Dense(1, activation="sigmoid"))
    trend_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Reshape for LSTM
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    trend_model.fit(X_train_lstm, y_train, epochs=5, batch_size=32, verbose=1)

    return trend_model


def train_volatility_model(X_train, y_train):
    """
    Train a Gradient Boosting model for predicting volatility.
    """
    volatility_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.2, max_depth=3, random_state=42)
    volatility_model.fit(X_train, y_train)

    return volatility_model


def retrain_models_if_needed():
    global last_training_time, trend_model, volatility_model, scaler
    current_time = time.time()
    if current_time - last_training_time >= RETRAIN_INTERVAL:
        print("Retraining models with updated data...")

        # Load local data
        features, trend_target, volatility_target, new_scaler = load_data()

        # Load online data and combine if available
        online_features, online_trend_target, online_volatility_target = load_online_data()
        if online_features is not None and not online_features.empty:
            features = pd.concat([features, online_features], ignore_index=True)
            trend_target = pd.concat([trend_target, online_trend_target], ignore_index=True)
            volatility_target = pd.concat([volatility_target, online_volatility_target], ignore_index=True)

        # Load trade log data and combine if available
        trade_log_features, trade_log_target = load_trade_log_data()
        if trade_log_features is not None and len(trade_log_features) > 0:
            # Ensure compatibility in the shape of features and targets
            if len(trade_log_features.columns) == len(features.columns):
                features = pd.concat([features, trade_log_features], ignore_index=True)
                trend_target = pd.concat([trend_target, trade_log_target], ignore_index=True)

        # Combine all data into a single DataFrame and drop NaN values
        combined_df = pd.concat([features, trend_target, volatility_target], axis=1)
        combined_df.dropna(inplace=True)

        # Check if there are enough samples to proceed
        if len(combined_df) == 0:
            print("No data available for training. Skipping retraining.")
            return

        # Split data back into features and targets
        features = combined_df[["MA3", "MA5", "MA15", "RSI", "Volatility"]]
        trend_target = combined_df.iloc[:, -2]
        volatility_target = combined_df.iloc[:, -1]

        # Split for training and testing
        X_train, X_test, y_train_trend, y_test_trend = train_test_split(features, trend_target, test_size=0.2, random_state=42)
        _, _, y_train_volatility, y_test_volatility = train_test_split(features, volatility_target, test_size=0.2, random_state=42)

        # Train the trend model (LSTM)
        with model_lock:
            trend_model = train_trend_model(X_train, y_train_trend)

        # Train the volatility model (Gradient Boosting)
        with model_lock:
            volatility_model = train_volatility_model(X_train, y_train_volatility)

        # Save models and scaler
        with model_lock:
            trend_model.save("models/lstm_trend_predictor_scalping.keras")  # Save as .keras
            joblib.dump(volatility_model, "models/volatility_predictor_scalping.pkl")
            joblib.dump(new_scaler, "models/scaler.pkl")
        
        scaler = new_scaler
        print("Model retraining and saving complete.")

        # Update last training time
        last_training_time = current_time


if __name__ == "__main__":
    # Initial training or loading of models
    retrain_models_if_needed()

    # Main loop to keep retraining models periodically
    # while True:
    #     retrain_models_if_needed()
    #     time.sleep(300)  # Check every 5 minutes if retraining is needed

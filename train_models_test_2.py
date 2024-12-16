# train_models.py - Refactored to OOP and Separate Trade Log Model

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
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Directory containing scalping data
data_directory = "data"  # Folder where data files are stored

class ModelManager:
    def __init__(self, model_path, scaler_path):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        try:
            if self.model_path.endswith('.keras'):
                self.model = load_model(self.model_path)
            else:
                self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print(f"Model loaded successfully from {self.model_path}.")
        except Exception as e:
            print(f"Could not load model: {e}")
            self.model, self.scaler = None, None

    def save_model(self):
        try:
            if self.model_path.endswith('.keras'):
                self.model.save(self.model_path)
            else:
                joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            print(f"Model saved successfully to {self.model_path}.")
        except Exception as e:
            print(f"Could not save model: {e}")

class DataLoader:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def load_data(self, target_column):
        all_data = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(pd.read_csv, os.path.join(self.data_directory, file_name))
                       for file_name in os.listdir(self.data_directory) if file_name.endswith("_scalping_data.csv")]
            for future in futures:
                try:
                    data = future.result()
                    all_data.append(data)
                except Exception as e:
                    print(f"Error loading data file: {e}")

        if not all_data:
            raise ValueError("No data files could be loaded.")

        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.dropna(subset=["MA3", "MA5", "MA15", "RSI", "Volatility"], how='all', inplace=True)

        features = combined_data[["MA3", "MA5", "MA15", "RSI", "Volatility"]]
        target = combined_data[target_column]

        imputer = SimpleImputer(strategy='mean')
        features_imputed = imputer.fit_transform(features)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_imputed)

        features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
        return features_scaled_df, target, scaler

    def load_trade_log_data(self, trade_log_file):
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

class TrendModelManager(ModelManager):
    def train_model(self, X_train, y_train):
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
        trend_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)
        self.model = trend_model

    def evaluate_model(self, X_test, y_test):
        if self.model:
            X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            loss, accuracy = self.model.evaluate(X_test_lstm, y_test, verbose=0)
            print(f"Trend Model - Loss: {loss}, Accuracy: {accuracy}")

class VolatilityModelManager(ModelManager):
    def train_model(self, X_train, y_train):
        volatility_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        volatility_model.fit(X_train, y_train)
        self.model = volatility_model

    def evaluate_model(self, X_test, y_test):
        if self.model:
            predictions = self.model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            print(f"Volatility Model - Mean Squared Error: {mse}")

class TradeLogModelManager(ModelManager):
    def train_model(self, X_train, y_train):
        trade_log_model = Sequential()
        trade_log_model.add(Input(shape=(X_train.shape[1], 1)))
        trade_log_model.add(LSTM(64, return_sequences=True))
        trade_log_model.add(Dropout(0.3))
        trade_log_model.add(LSTM(64))
        trade_log_model.add(Dropout(0.3))
        trade_log_model.add(Dense(1, activation="sigmoid"))
        trade_log_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Reshape for LSTM
        X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        trade_log_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)
        self.model = trade_log_model

    def evaluate_model(self, X_test, y_test):
        if self.model:
            X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            loss, accuracy = self.model.evaluate(X_test_lstm, y_test, verbose=0)
            print(f"Trade Log Model - Loss: {loss}, Accuracy: {accuracy}")

class RetrainManager:
    def __init__(self, trend_model_manager, volatility_model_manager, trade_log_model_manager, data_loader, retrain_interval=60 * 60):
        self.trend_model_manager = trend_model_manager
        self.volatility_model_manager = volatility_model_manager
        self.trade_log_model_manager = trade_log_model_manager
        self.data_loader = data_loader
        self.retrain_interval = retrain_interval
        self.last_training_time = 0

    def retrain_models_if_needed(self):
        current_time = time.time()
        if current_time - self.last_training_time >= self.retrain_interval:
            print("Retraining models with updated data...")
            self.retrain_trend_model()
            self.retrain_volatility_model()
            self.retrain_trade_log_model()
            self.last_training_time = current_time

    def retrain_trend_model(self):
        features, trend_target, scaler = self.data_loader.load_data("Trend")
        features = features.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(features, trend_target, test_size=0.2, random_state=42)
        self.trend_model_manager.train_model(X_train, y_train)
        self.trend_model_manager.evaluate_model(X_test, y_test)
        self.trend_model_manager.scaler = scaler
        self.trend_model_manager.save_model()

    def retrain_volatility_model(self):
        features, volatility_target, _ = self.data_loader.load_data("Volatility")
        features = features.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(features, volatility_target, test_size=0.2, random_state=42)
        self.volatility_model_manager.train_model(X_train, y_train)
        self.volatility_model_manager.evaluate_model(X_test, y_test)
        self.volatility_model_manager.save_model()

    def retrain_trade_log_model(self):
        features, trade_target = self.data_loader.load_trade_log_data("trades_log.csv")
        if features is None or trade_target is None:
            print("No trade log data available for retraining.")
            return
        features = features.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(features, trade_target, test_size=0.2, random_state=42)
        self.trade_log_model_manager.train_model(X_train, y_train)
        self.trade_log_model_manager.evaluate_model(X_test, y_test)
        self.trade_log_model_manager.save_model()

if __name__ == "__main__":
    data_loader = DataLoader(data_directory)
    trend_model_manager = TrendModelManager("models/lstm_trend_predictor_scalping.keras", "models/scaler.pkl")
    volatility_model_manager = VolatilityModelManager("models/volatility_predictor_scalping.pkl", "models/scaler.pkl")
    trade_log_model_manager = TradeLogModelManager("models/trade_log_scalping.keras", "models/scaler_trade_log.pkl")
    retrain_manager = RetrainManager(trend_model_manager, volatility_model_manager, trade_log_model_manager, data_loader)

    # Initial training or loading of models
    retrain_manager.retrain_models_if_needed()

    # Main loop to keep retraining models periodically
    # while True:
    #     retrain_manager.retrain_models_if_needed()
    #     time.sleep(300)  # Check every 5 minutes if retraining is needed

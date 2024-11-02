import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# Directory containing scalping data
data_directory = "data"  # Folder where data files are stored

def load_data():
    """
    Load and preprocess data from all CSV files in the data directory.
    """
    all_data = []
    for file_name in os.listdir(data_directory):
        if file_name.endswith("_scalping_data.csv"):
            file_path = os.path.join(data_directory, file_name)
            data = pd.read_csv(file_path)
            all_data.append(data)
            print(f"Loaded data from {file_name}")
    
    # Concatenate all data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)
    features = combined_data[["MA3", "MA5", "MA15", "RSI", "Volatility"]]
    trend_target = combined_data["Trend"]
    volatility_target = combined_data["Volatility"]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, trend_target, volatility_target, scaler

# Training models
def train_trend_model(X_train, y_train):
    """
    Train LSTM model for predicting short-term price trend.
    """
    trend_model = Sequential()
    trend_model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    trend_model.add(Dropout(0.2))
    trend_model.add(LSTM(64))
    trend_model.add(Dropout(0.2))
    trend_model.add(Dense(1, activation="sigmoid"))
    trend_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Reshape for LSTM
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    trend_model.fit(X_train_lstm, y_train, epochs=10, batch_size=64, verbose=1)

    return trend_model

def train_volatility_model(X_train, y_train):
    """
    Train a Gradient Boosting model for predicting volatility.
    """
    volatility_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    volatility_model.fit(X_train, y_train)

    return volatility_model

if __name__ == "__main__":
    # Load and split data for training
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

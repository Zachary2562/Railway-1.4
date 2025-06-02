import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import ta
import os

st.set_page_config(page_title="AI Forecast App", layout="wide")
st.title("📈 AI Forecast App By Zachary2562")

# Ticker Selection
if os.path.exists("tickers.txt"):
    ticker_list = open("tickers.txt").read().splitlines()
else:
    ticker_list = ["AAPL", "GOOG", "MSFT", "TSLA", "BTC-USD", "ETH-USD"]

selected_ticker = st.sidebar.selectbox("Select an option", ticker_list)
custom_ticker = st.sidebar.text_input("🔎 Search Yahoo Finance (e.g., NVDA, SPY, GOLD)", "")
ticker = custom_ticker.upper() if custom_ticker else selected_ticker

# Load and preprocess data
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2010-01-01")
    df.dropna(inplace=True)
    return df

df = load_data(ticker)
st.subheader(f"📊 Historical Close Price: {ticker}")
st.line_chart(df["Close"])

# Add Technical Indicators
df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"].squeeze()).rsi()
df["MACD"] = ta.trend.MACD(close=df["Close"]).macd()
st.line_chart(df[["RSI", "MACD"]])

# Prophet Forecast
st.subheader("🔮 Prophet Forecast")
df_prophet = df.reset_index()[["Date", "Close"]]
df_prophet.columns = ["ds", "y"]
model = Prophet()
model.fit(df_prophet)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Optional LSTM Forecast
enable_lstm = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    enable_lstm = True
except ImportError:
    st.warning("TensorFlow not available. LSTM forecast disabled.")

if enable_lstm:
    st.subheader("🔁 LSTM Forecast (Experimental)")
    data = df.filter(["Close"])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * 0.8))

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:training_data_len]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model_lstm = Sequential()
    model_lstm.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(64, return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')

    model_lstm.fit(x_train, y_train, batch_size=32, epochs=20)

    test_data = scaled_data[training_data_len - 60:]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model_lstm.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    true_values = scaler.inverse_transform(dataset[training_data_len:])
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    st.write(f"🔍 LSTM Forecast RMSE: {rmse:.2f}")

    valid = data[training_data_len:]
    valid["Predictions"] = predictions
    st.line_chart(valid[["Close", "Predictions"]])

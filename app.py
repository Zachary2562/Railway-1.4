import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
import ta

st.set_page_config(page_title="AI Forecast App", layout="wide")
st.title("📈 AI Forecast App By Zachary2562")

# Sidebar for stock selection and accuracy mode
ticker = st.sidebar.text_input("🔎 Search Yahoo Finance (e.g., AAPL, TSLA)", value="AAPL")
accuracy_mode = st.sidebar.selectbox("Forecast Accuracy Mode", ["Normal (Fast)", "High (Slower)"])
epochs = 50 if accuracy_mode == "High (Slower)" else 20

# Fetch data
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2010-01-01")
    df.dropna(inplace=True)
    return df

df = load_data(ticker)

# Display raw data
st.subheader(f"{ticker} Historical Data")
st.dataframe(df.tail())

# --- Calculate Technical Indicators ---
close_prices = df["Close"].astype(float)

# Safely compute indicators
try:
    df["RSI"] = ta.momentum.RSIIndicator(close=close_prices).rsi().squeeze()
except Exception as e:
    st.error(f"RSI calculation failed: {e}")

try:
    df["MACD"] = ta.trend.MACD(close=close_prices).macd().squeeze()
except Exception as e:
    st.error(f"MACD calculation failed: {e}")

# Plot indicators if available
indicators_to_plot = [col for col in ["RSI", "MACD"] if col in df.columns]
if indicators_to_plot:
    st.subheader("Technical Indicators")
    st.line_chart(df[indicators_to_plot])
else:
    st.warning("RSI and MACD indicators could not be computed.")

# Prophet Forecasting
st.subheader("📅 Prophet Forecast")

df_reset = df.reset_index()
df_prophet = df_reset[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_prophet)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

fig1 = model.plot(forecast)
st.pyplot(fig1)

# Show forecasted data
st.subheader("Forecasted Data")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

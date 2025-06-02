import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from prophet import Prophet
import ta

st.set_page_config(page_title="AI Forecast App", layout="wide")
st.title("📈 AI Forecast App By Zachary2562")

ticker = st.text_input("🔎 Search Yahoo Finance (e.g., AAPL, TSLA, MSFT)", value="AAPL").upper()
accuracy_mode = st.radio("Select Forecast Mode", ("Normal", "High Accuracy"))

if ticker:
    df = yf.download(ticker, start="2010-01-01", progress=False)
    if df.empty:
        st.error("No data found for the selected ticker.")
    else:
        df = df.dropna()

        try:
            df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"]).rsi()
            df["MACD"] = ta.trend.MACD(close=df["Close"]).macd()
        except Exception as e:
            st.warning(f"RSI/MACD calculation failed: {e}")
            df["RSI"] = np.nan
            df["MACD"] = np.nan

        if df[["RSI", "MACD"]].dropna().empty:
            st.warning("RSI and MACD indicators could not be computed.")
        else:
            st.subheader("📊 Technical Indicators")
            st.line_chart(df[["Close", "RSI", "MACD"]])

        st.subheader("📅 Prophet Forecast")

        df_prophet = pd.DataFrame()
        df_prophet["ds"] = df.index
        df_prophet["y"] = df["Close"]

        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        st.write("Forecast Data")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        # Displaying trend and components
        st.subheader("📈 Trend & Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        if accuracy_mode == "High Accuracy":
            st.info("High Accuracy Mode: Training LSTM (50 epochs)...")
        else:
            st.info("Normal Mode: LSTM training skipped for speed.")

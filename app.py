# app.py

import os

# ── Disable oneDNN optimizations and silence TensorFlow logs ──────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import requests

# ── TA technical indicators (restored imports) ───────────────────────
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

# ── Prophet ──────────────────────────────────────────────────────────
from prophet import Prophet

# ── TensorFlow / Keras (LSTM) ────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# ── Fuelfinance (basic risk analytics) ───────────────────────────────
import fuelfinance as ff


# ------------------------------- Load Data (Tiingo) --------------------------------
@st.cache_data(show_spinner=False)
def load_data(ticker_symbol):
    """
    Fetch up to the last 10 years of daily OHLCV from Tiingo.
    Returns (df, status):
      - status == "ok":    df contains Tiingo data (date/open/high/low/close/volume)
      - status == "error": network issue, missing/invalid API key, or no data
    """
    import pandas as pd
    import requests
    import os

    t = ticker_symbol.strip().upper()
    if not t:
        return pd.DataFrame(), "error"

    TIINGO_KEY = os.getenv("TIINGO_API_KEY", "")
    if not TIINGO_KEY:
        return pd.DataFrame(), "error"

    ten_years_ago = (pd.Timestamp.today() - pd.DateOffset(years=10)).date().isoformat()
    url = (
        f"https://api.tiingo.com/tiingo/daily/{t}/prices"
        f"?startDate={ten_years_ago}&format=json&token={TIINGO_KEY}"
    )

    try:
        r = requests.get(url, timeout=10)
        data = r.json()
    except Exception as e:
        print(f"[Tiingo Fetch Error] {t}: {e}")
        return pd.DataFrame(), "error"

    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(), "error"

    df = pd.DataFrame(data).rename(columns={
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df, "ok"


# ------------------------ UI SETUP ---------------------------------------
st.set_page_config(page_title="AI Forecast App", layout="wide")
st.title("📈 AI Forecast App by Zachary2562")

# ─── Sidebar: Mode Selection ──────────────────────────────────────────
st.sidebar.header("⚙️ Mode Selection")
mode = st.sidebar.radio(
    "Choose mode",
    ["Regular", "Advanced"],
    index=0,
    help="Regular: use default hyperparameters. Advanced: tweak everything manually."
)

# ─── Sidebar: Ticker Dropdown & Search ─────────────────────────────────
st.sidebar.header("🔍 Select or Search a Ticker")
PRELOADED_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "FB", "TSLA", "NVDA", "BRK-B", "JPM", "V",
    "UNH", "HD", "PG", "MA", "BAC", "XOM", "DIS", "VZ", "ADBE", "NFLX",
    "INTC", "PFE", "KO", "CMCSA", "PEP", "T", "CSCO", "ABT", "CVX", "NKE",
    "MRK", "WMT", "CRM", "ORCL", "ACN", "MCD", "DHR", "COST", "MDT", "LLY",
    "TXN", "QCOM", "NEE", "AMGN", "HON", "IBM", "BMY", "AVGO", "UNP", "SBUX"
]
selected_dropdown = st.sidebar.selectbox("Pick one:", PRELOADED_TICKERS)
custom_ticker = st.sidebar.text_input(
    "Or enter a custom ticker (e.g. AAPL)",
    value="",
    help="Type a symbol not in the dropdown."
)
ticker = custom_ticker.strip().upper() if custom_ticker else selected_dropdown

# ─── Sidebar: Date Range (label only) ─────────────────────────────────
START_DATE = "2010-01-01"
END_DATE = datetime.date.today().strftime("%Y-%m-%d")

# ─── Sidebar: LSTM Hyperparameters ────────────────────────────────────
if mode == "Regular":
    n_lstm_layers = 1
    lstm_units    = 32    # reduced from 64
    dropout_rate  = 0.2
    batch_size    = 16    # reduced from 32
    epochs        = 10    # reduced from 25
    st.sidebar.write("Using **Regular** mode defaults (no tuning):")
    st.sidebar.write(f"• LSTM layers: {n_lstm_layers}")
    st.sidebar.write(f"• Units per layer: {lstm_units}")
    st.sidebar.write(f"• Dropout rate: {dropout_rate}")
    st.sidebar.write(f"• Batch size: {batch_size}")
    st.sidebar.write(f"• Epochs: {epochs}")
else:
    st.sidebar.subheader("📊 LSTM Hyperparameters (Advanced)")
    n_lstm_layers = st.sidebar.selectbox("Number of LSTM layers", [1, 2], index=0)
    lstm_units    = st.sidebar.slider("Units per layer", min_value=32, max_value=256, value=32, step=32)
    dropout_rate  = st.sidebar.slider("Dropout rate", min_value=0.0, max_value=0.5, value=0.2, step=0.1)
    batch_size    = st.sidebar.selectbox("Batch size", [8, 16, 32], index=1)
    epochs        = st.sidebar.number_input("Epochs", min_value=1, max_value=50, value=10, step=1)

# ─── Sidebar: Forecast Horizon ────────────────────────────────────────
st.sidebar.subheader("📆 Forecast Settings")
forecast_years = st.sidebar.slider("Years to forecast into future", 1, 5, 1)
forecast_days  = forecast_years * 252  # Approximate trading days per year

# ─── Sidebar: Run Button ─────────────────────────────────────────────
run_button = st.sidebar.button("▶ Run Forecast")


# ─── Main: Run Forecast Pipeline ─────────────────────────────────────
if run_button:

    # 1️⃣ Load data via Tiingo (last 10 years)
    data_load_state = st.text("Loading data...")
    df, status = load_data(ticker)

    if status == "error":
        st.error(
            f"❌ Could not fetch data for “{ticker}”.  \n"
            "– Make sure TIINGO_API_KEY is set correctly in Railway Variables.  \n"
            "– Check that the symbol exists on Tiingo."
        )
        st.stop()

    st.success(f"Data loaded successfully! ({len(df)} rows)")

    # 2️⃣ Compute technical indicators
    st.subheader("🔧 Computing Technical Indicators")
    df_ind = df.copy()
    df_ind["rsi"] = RSIIndicator(close=df_ind["close"], window=14).rsi()
    macd = MACD(close=df_ind["close"], window_slow=26, window_fast=12, window_sign=9)
    df_ind["macd"] = macd.macd_diff()
    bb = BollingerBands(close=df_ind["close"], window=20, window_dev=2)
    df_ind["bb_hband"] = bb.bollinger_hband()
    df_ind["bb_lband"] = bb.bollinger_lband()
    df_ind["bb_mavg"]  = bb.bollinger_mavg()
    df_ind["vwap"] = VolumeWeightedAveragePrice(
        high=df_ind["high"],
        low=df_ind["low"],
        close=df_ind["close"],
        volume=df_ind["volume"],
        window=14
    ).volume_weighted_average_price()

    df_ind.dropna(inplace=True)
    st.success(f"Technical indicators computed. ({len(df_ind):,} rows)")

    # 3️⃣ Plot historical close + RSI + MACD instead of table
    st.subheader("📊 Historical Close & Indicators")
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # a) Close price
    axs[0].plot(df_ind["date"], df_ind["close"], color="black", label="Close Price")
    axs[0].set_ylabel("Close")
    axs[0].legend()

    # b) RSI
    axs[1].plot(df_ind["date"], df_ind["rsi"], color="blue", label="RSI (14)")
    axs[1].axhline(70, color="red", linestyle="--", linewidth=0.5)
    axs[1].axhline(30, color="green", linestyle="--", linewidth=0.5)
    axs[1].set_ylabel("RSI")
    axs[1].legend()

    # c) MACD diff
    axs[2].plot(df_ind["date"], df_ind["macd"], color="purple", label="MACD Diff")
    axs[2].axhline(0, color="gray", linestyle="--", linewidth=0.5)
    axs[2].set_ylabel("MACD Diff")
    axs[2].legend()

    axs[2].set_xlabel("Date")
    st.pyplot(fig)

    # 4️⃣ Define features (X) and target (y)
    st.subheader("📈 Preparing Data for Modeling")
    feature_cols = [
        "rsi","macd","bb_mavg","bb_hband","bb_lband","vwap",
        "open","high","low","volume"
    ]
    X_all = df_ind[feature_cols].values
    y_all = df_ind["close"].values

    # 5️⃣ Train-test split (last 10% as test set)
    split_idx = int(len(X_all) * 0.9)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    dates_test      = df_ind["date"].values[split_idx:]

    st.write(f"• Training samples: {X_train.shape[0]}")
    st.write(f"• Testing samples:  {X_test.shape[0]}")

    # 6️⃣ Prophet forecasting
    st.subheader("🔮 Prophet Forecast")
    prophet_df = df_ind[["date","close"]].rename(columns={"date": "ds", "close": "y"})
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(prophet_df)
    future   = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)
    prophet_forecast_df = forecast[["ds","yhat"]].tail(forecast_days).copy()
    prophet_forecast_df.rename(columns={"ds": "date", "yhat": "prophet_yhat"}, inplace=True)
    st.success(f"Prophet model trained and forecast generated for {forecast_days} trading days.")

    # 7️⃣ LSTM forecasting
    st.subheader("🤖 LSTM Forecast")

    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X_all)
    y_scaled = target_scaler.fit_transform(y_all.reshape(-1,1))

    LOOKBACK = 30  # reduced from 60
    def create_sequences(X, y, lookback):
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i - lookback : i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    X_seq_all, y_seq_all = create_sequences(X_scaled, y_scaled, LOOKBACK)

    split_seq_idx     = int(len(X_seq_all) * 0.9)
    X_seq_train       = X_seq_all[:split_seq_idx]
    X_seq_test        = X_seq_all[split_seq_idx:]
    y_seq_train       = y_seq_all[:split_seq_idx]
    y_seq_test        = y_seq_all[split_seq_idx:]

    model = Sequential()
    for layer_idx in range(n_lstm_layers):
        return_seq = (layer_idx < n_lstm_layers - 1)
        if layer_idx == 0:
            model.add(
                LSTM(
                    units=lstm_units,
                    return_sequences=return_seq,
                    input_shape=(LOOKBACK, X_seq_train.shape[2])
                )
            )
        else:
            model.add(
                LSTM(
                    units=lstm_units,
                    return_sequences=return_seq
                )
            )
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    st.write(f"• LSTM architecture: {n_lstm_layers} layer(s) × {lstm_units} units, dropout={dropout_rate}")
    st.write(f"• Batch size: {batch_size}, Epochs: {epochs}")

    try:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
        history = model.fit(
            X_seq_train,
            y_seq_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        st.success("LSTM model trained.")
    except Exception as e:
        st.error(f"⚠️ LSTM training failed: {e}")
        st.stop()

    # Forecast future with LSTM (iterative)
    last_sequence = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, len(feature_cols))
    lstm_forecasts_scaled = []
    cur_seq = last_sequence.copy()

    for _ in range(forecast_days):
        pred_scaled = model.predict(cur_seq, verbose=0).flatten()[0]
        lstm_forecasts_scaled.append(pred_scaled)
        last_row = cur_seq[0, -1, :].copy()
        last_row[0] = pred_scaled  # assuming “close” is first feature
        next_seq = np.concatenate([cur_seq[0, 1:], last_row.reshape(1, -1)], axis=0)
        cur_seq = next_seq.reshape(1, LOOKBACK, len(feature_cols))

    lstm_forecasts = target_scaler.inverse_transform(
        np.array(lstm_forecasts_scaled).reshape(-1, 1)
    ).flatten()

    # 8️⃣ Combine Prophet & LSTM forecasts
    prophet_vals     = prophet_forecast_df["prophet_yhat"].values[:forecast_days]
    ensemble_forecast = (prophet_vals + lstm_forecasts) / 2

    # 9️⃣ Backtesting on test set
    st.subheader("📊 Backtesting Metrics (Test Set)")
    prop_test_vals = forecast["yhat"].values[-(len(y_test) + forecast_days):-forecast_days]
    prop_mape = mean_absolute_percentage_error(y_test, prop_test_vals)
    prop_rmse = np.sqrt(mean_squared_error(y_test, prop_test_vals))

    y_lstm_test_scaled = model.predict(X_seq_test, verbose=0).flatten()
    y_lstm_test        = target_scaler.inverse_transform(y_lstm_test_scaled.reshape(-1,1)).flatten()
    y_lstm_true        = y_all[LOOKBACK + split_idx : LOOKBACK + split_idx + len(y_lstm_test)]
    lstm_mape = mean_absolute_percentage_error(y_lstm_true, y_lstm_test)
    lstm_rmse = np.sqrt(mean_squared_error(y_lstm_true, y_lstm_test))

    st.write(f"• Prophet MAPE: {prop_mape:.2%}, RMSE: {prop_rmse:.3f}")
    st.write(f"• LSTM    MAPE: {lstm_mape:.2%}, RMSE: {lstm_rmse:.3f}")

    # 10️⃣ Plot final forecasts vs historical
    st.subheader("📈 Forecast Comparison Plot")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(df_ind["date"], df_ind["close"], label="Historical Close", color="black")
    future_dates = prophet_forecast_df["date"]
    ax2.plot(future_dates, prophet_vals,      label="Prophet Forecast", linestyle="--")
    ax2.plot(future_dates, lstm_forecasts,    label="LSTM Forecast",    linestyle=":")
    ax2.plot(future_dates, ensemble_forecast, label="Ensemble Forecast", linewidth=2)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()
    st.pyplot(fig2)

    # 11️⃣ Fuelfinance risk analytics
    st.subheader("💼 Fuelfinance Risk Analytics")
    hist    = df.set_index("date")["close"]
    returns = hist.pct_change().dropna()
    sharpe        = ff.sharpe_ratio(returns, rf=0.0, period="daily")
    volatility_nm = ff.volatility(returns, period="daily")
    max_dd        = ff.max_drawdown(returns)
    st.write(f"• Sharpe Ratio: {sharpe:.4f}")
    st.write(f"• Volatility: {volatility_nm:.4%}")
    st.write(f"• Max Drawdown: {max_dd:.2%}")

    # 12️⃣ Download forecasts
    st.subheader("🗒️ Download Data & Forecasts")
    export_df = pd.DataFrame({
        "date":              future_dates,
        "prophet_forecast":  prophet_vals,
        "lstm_forecast":     lstm_forecasts,
        "ensemble_forecast": ensemble_forecast
    })
    export_csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download future forecasts as CSV",
        data=export_csv,
        file_name=f"{ticker}_future_forecast.csv",
        mime="text/csv"
    )

    st.success("✅ Forecasting complete!")

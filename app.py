# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt

# --- Technical indicators ---
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

# --- Prophet ---
from prophet import Prophet

# --- LSTM (TensorFlow/Keras) ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# --- Fuelfinance (basic risk analytics) ---
import fuelfinance as ff

# ---------------------------- UI SETUP ----------------------------
st.set_page_config(page_title="AI Forecast App", layout="wide")
st.title("📈 AI Forecast App by Zachary2562")

# ─── Sidebar: Mode Selection ────────────────────────────────────────
st.sidebar.header("⚙️ Mode Selection")
mode = st.sidebar.radio(
    "Choose mode",
    ["Regular", "Advanced"],
    index=0,
    help="Regular: use default hyperparameters. Advanced: tweak everything manually."
)

# ─── Sidebar: Ticker Dropdown ───────────────────────────────────────
st.sidebar.header("🔍 Select a Ticker")
# Preloaded list of ~50 popular tickers
PRELOADED_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "FB", "TSLA", "NVDA", "BRK-B", "JPM", "V", 
    "UNH", "HD", "PG", "MA", "BAC", "XOM", "DIS", "VZ", "ADBE", "NFLX",
    "INTC", "PFE", "KO", "CMCSA", "PEP", "T", "CSCO", "ABT", "CVX", "NKE",
    "MRK", "WMT", "CRM", "ORCL", "ACN", "MCD", "DHR", "COST", "MDT", "LLY",
    "TXN", "QCOM", "NEE", "AMGN", "HON", "IBM", "BMY", "AVGO", "UNP", "SBUX"
]
ticker = st.sidebar.selectbox("Pick one:", PRELOADED_TICKERS)

# ─── Sidebar: Date Range (fixed) ────────────────────────────────────
START_DATE = "2010-01-01"
END_DATE = datetime.date.today().strftime("%Y-%m-%d")

# ─── Sidebar: LSTM Hyperparameters ─────────────────────────────────
# We’ll either use defaults (Regular) or show all controls (Advanced).
if mode == "Regular":
    n_lstm_layers = 1
    lstm_units    = 64
    dropout_rate  = 0.2
    batch_size    = 32
    epochs        = 25
    st.sidebar.write("Using **Regular** mode defaults (no tuning):")
    st.sidebar.write(f"• LSTM layers: {n_lstm_layers}")
    st.sidebar.write(f"• Units per layer: {lstm_units}")
    st.sidebar.write(f"• Dropout rate: {dropout_rate}")
    st.sidebar.write(f"• Batch size: {batch_size}")
    st.sidebar.write(f"• Epochs: {epochs}")
else:
    st.sidebar.subheader("📊 LSTM Hyperparameters (Advanced)")
    n_lstm_layers = st.sidebar.selectbox("Number of LSTM layers", [1, 2], index=0)
    lstm_units    = st.sidebar.slider("Units per layer", min_value=32, max_value=256, value=64, step=32)
    dropout_rate  = st.sidebar.slider("Dropout rate", min_value=0.0, max_value=0.5, value=0.2, step=0.1)
    batch_size    = st.sidebar.selectbox("Batch size", [16, 32, 64], index=1)
    epochs        = st.sidebar.number_input("Epochs", min_value=5, max_value=100, value=25, step=5)

# ─── Sidebar: Forecast Horizon ──────────────────────────────────────
st.sidebar.subheader("📆 Forecast Settings")
forecast_years = st.sidebar.slider("Years to forecast into future", 1, 5, 1)
forecast_days  = forecast_years * 252  # Approximate trading days per year

# ─── Sidebar: Run Button ───────────────────────────────────────────
run_button = st.sidebar.button("▶ Run Forecast")

if run_button:

    @st.cache_data(show_spinner=False)
    def load_data(ticker_symbol):
        df = yf.download(
            ticker_symbol, 
            start=START_DATE, 
            end=END_DATE, 
            progress=False
        )
        df.reset_index(inplace=True)
        df.rename(
            columns={
                "Date":   "date",
                "Close":  "close",
                "Open":   "open",
                "High":   "high",
                "Low":    "low",
                "Volume": "volume"
            },
            inplace=True
        )
        return df

    # 1️⃣ Load historical data
    data_load_state = st.text("Loading data...")
    df = load_data(ticker)
    data_load_state.text(f"Data loaded successfully! ({len(df)} rows)")

    if df.empty:
        st.error(f"No data found for ticker “{ticker}”. Please check the symbol and try again.")
        st.stop()

    # 2️⃣ Compute technical indicators
    st.subheader("🔧 Computing Technical Indicators")
    df_ind = df.copy()
    # RSI
    df_ind["rsi"] = RSIIndicator(close=df_ind["close"], window=14).rsi()
    # MACD
    macd = MACD(close=df_ind["close"], window_slow=26, window_fast=12, window_sign=9)
    df_ind["macd"] = macd.macd_diff()
    # Bollinger Bands (20-day, 2 std)
    bb = BollingerBands(close=df_ind["close"], window=20, window_dev=2)
    df_ind["bb_hband"] = bb.bollinger_hband()
    df_ind["bb_lband"] = bb.bollinger_lband()
    df_ind["bb_mavg"]  = bb.bollinger_mavg()
    # VWAP (requires high, low, close, volume)
    df_ind["vwap"] = VolumeWeightedAveragePrice(
        high=df_ind["high"],
        low=df_ind["low"],
        close=df_ind["close"],
        volume=df_ind["volume"],
        window=14
    ).volume_weighted_average_price()

    df_ind.dropna(inplace=True)
    st.success(f"Technical indicators computed. ({len(df_ind):,} rows)")

    # 3️⃣ Show a preview
    st.dataframe(
        df_ind[["date","open","high","low","close","volume",
                "rsi","macd","bb_mavg","bb_hband","bb_lband","vwap"]].tail(10)
    )

    # 4️⃣ Define features (X) and target (y)
    st.subheader("📈 Preparing Data for Modeling")
    feature_cols = [
        "rsi","macd","bb_mavg","bb_hband","bb_lband","vwap",
        "open","high","low","volume"
    ]
    X_all = df_ind[feature_cols].values
    y_all = df_ind["close"].values  # 1D array of closing prices

    # 5️⃣ Train-test split (last 10% as test)
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

    # Scale features and target
    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X_all)
    y_scaled = target_scaler.fit_transform(y_all.reshape(-1,1))

    # Create sequences (lookback window)
    LOOKBACK = 60  # last 60 days to predict next
    def create_sequences(X, y, lookback):
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i - lookback : i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    X_seq_all, y_seq_all = create_sequences(X_scaled, y_scaled, LOOKBACK)

    # Re-split into train/test sequences
    split_seq_idx     = int(len(X_seq_all) * 0.9)
    X_seq_train       = X_seq_all[:split_seq_idx]
    X_seq_test        = X_seq_all[split_seq_idx:]
    y_seq_train       = y_seq_all[:split_seq_idx]
    y_seq_test        = y_seq_all[split_seq_idx:]

    # Build LSTM model
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

    # Train LSTM with EarlyStopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
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

    # Forecast future with LSTM
    last_sequence = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, len(feature_cols))
    lstm_forecasts_scaled = []
    cur_seq = last_sequence.copy()

    for _ in range(forecast_days):
        pred_scaled = model.predict(cur_seq, verbose=0).flatten()[0]
        lstm_forecasts_scaled.append(pred_scaled)
        # Build next sequence: reuse last features, replace close with pred_scaled
        last_row = cur_seq[0, -1, :].copy()  # shape=(features,)
        last_row[0] = pred_scaled  # assuming “close” is the first feature; adjust if different
        next_seq = np.concatenate([cur_seq[0, 1:], last_row.reshape(1, -1)], axis=0)
        cur_seq = next_seq.reshape(1, LOOKBACK, len(feature_cols))

    lstm_forecasts = target_scaler.inverse_transform(
        np.array(lstm_forecasts_scaled).reshape(-1, 1)
    ).flatten()

    # 8️⃣ Combine Prophet & LSTM forecasts (average)
    prophet_vals     = prophet_forecast_df["prophet_yhat"].values[:forecast_days]
    ensemble_forecast = (prophet_vals + lstm_forecasts) / 2

    # 9️⃣ Backtesting on test set
    st.subheader("📊 Backtesting Metrics (Test Set)")

    # Prophet test backfill
    prophet_test = forecast["yhat"].values[-(len(y_test) + forecast_days):-forecast_days]
    prophet_test_true = y_test
    prop_mape = mean_absolute_percentage_error(prophet_test_true, prophet_test)
    prop_rmse = np.sqrt(mean_squared_error(prophet_test_true, prophet_test))

    # LSTM test backfill
    y_lstm_test_scaled = model.predict(X_seq_test, verbose=0).flatten()
    y_lstm_test        = target_scaler.inverse_transform(y_lstm_test_scaled.reshape(-1,1)).flatten()
    y_lstm_true        = y_all[LOOKBACK + split_idx : LOOKBACK + split_idx + len(y_lstm_test)]
    lstm_mape = mean_absolute_percentage_error(y_lstm_true, y_lstm_test)
    lstm_rmse = np.sqrt(mean_squared_error(y_lstm_true, y_lstm_test))

    st.write(f"• Prophet MAPE (test): {prop_mape:.2%}, RMSE: {prop_rmse:.3f}")
    st.write(f"• LSTM    MAPE (test): {lstm_mape:.2%}, RMSE: {lstm_rmse:.3f}")

    # 10️⃣ Plot results
    st.subheader("📈 Forecast Comparison Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_ind["date"], df_ind["close"], label="Historical Close", color="black")
    future_dates = prophet_forecast_df["date"]
    ax.plot(future_dates, prophet_vals,      label="Prophet Forecast", linestyle="--")
    ax.plot(future_dates, lstm_forecasts,    label="LSTM Forecast",    linestyle=":")
    ax.plot(future_dates, ensemble_forecast, label="Ensemble Forecast", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # 11️⃣ Fuelfinance risk analytics
    st.subheader("💼 Fuelfinance Risk Analytics")
    hist          = df.set_index("date")["close"]
    returns       = hist.pct_change().dropna()
    sharpe        = ff.sharpe_ratio(returns, rf=0.0, period="daily")
    volatility_nm = ff.volatility(returns, period="daily")
    max_dd        = ff.max_drawdown(returns)
    st.write(f"• Sharpe Ratio (daily): {sharpe:.4f}")
    st.write(f"• Volatility (daily): {volatility_nm:.4%}")
    st.write(f"• Max Drawdown: {max_dd:.2%}")

    # 12️⃣ Data export options
    st.subheader("🗒️ Download Data & Forecasts")
    export_df = pd.DataFrame({
        "date":            future_dates,
        "prophet_forecast": prophet_vals,
        "lstm_forecast":    lstm_forecasts,
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

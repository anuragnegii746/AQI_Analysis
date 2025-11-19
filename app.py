import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.title("AQI Forecasting Dashboard")

# ---------- CONFIG ----------
# Use ABSOLUTE PATH to avoid any path confusion
DATA_PATH = r"C:\Users\Chetna Negi\OneDrive\Desktop\AQI_Forecasting\data\processed\merged_output_final.csv"
POLLUTANTS = ["pm25", "pm10", "o3", "no2", "so2", "co"]
N_LAGS = 7  # how many past days to use as features


# ---------- DATA LOADING & CLEANING ----------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Parse dates robustly
    df["date"] = pd.to_datetime(
        df["date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )
    df = df.dropna(subset=["date"])

    # Ensure pollutant columns are numeric
    for col in POLLUTANTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where all pollutants are NaN
    df = df.dropna(subset=POLLUTANTS, how="all")

    # Sort
    df = df.sort_values(["center", "date"])

    return df


try:
    df = load_data(DATA_PATH)
    st.success(f"Data loaded. Shape: {df.shape}")
except Exception as e:
    st.error("Error loading data. Check DATA_PATH in app.py")
    st.code(str(e))
    st.stop()

# ---------- SIDEBAR CONTROLS ----------
st.sidebar.header("Controls")

centers = sorted(df["center"].dropna().unique().tolist())
selected_center = st.sidebar.selectbox("Select center", centers)

available_pollutants = [p for p in POLLUTANTS if p in df.columns]
selected_pollutant = st.sidebar.selectbox("Select pollutant", available_pollutants)

min_date = df["date"].min()
max_date = df["date"].max()

date_range = st.sidebar.date_input(
    "Historical date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, date_range

forecast_horizon = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=3,
    max_value=30,
    value=7
)

st.sidebar.markdown("---")
st.sidebar.write("Model: RandomForest + lag features")


# ---------- FILTER DATA FOR SELECTION ----------
mask = (
    (df["center"] == selected_center) &
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
)
df_center = df.loc[mask, ["date", selected_pollutant]].dropna()
df_center = df_center.sort_values("date")

if df_center.empty:
    st.warning("No data for this center and date range. Try changing filters.")
    st.stop()

if len(df_center) < (N_LAGS + 10):
    st.warning("Not enough data points for forecasting. Try expanding the date range.")
    st.stop()


# ---------- FORECAST FUNCTION ----------
def make_forecast(series: pd.Series, horizon: int, n_lags: int = 7):
    # Ensure date index, regular daily frequency, forward fill
    series = series.sort_index()
    full_index = pd.date_range(start=series.index.min(), end=series.index.max(), freq="D")
    series = series.reindex(full_index)
    series = series.ffill()

    # Build lag features
    df_ts = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df_ts[f"lag_{lag}"] = df_ts["y"].shift(lag)

    df_model = df_ts.dropna()
    if len(df_model) < 20:
        raise ValueError("Not enough data after constructing lag features.")

    feature_cols = [f"lag_{lag}" for lag in range(1, n_lags + 1)]
    X = df_model[feature_cols]
    y = df_model["y"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    history = list(series.dropna())
    last_lags = history[-n_lags:]
    preds = []
    last_date = series.index.max()

    for _ in range(horizon):
        x_input = np.array(last_lags[-n_lags:]).reshape(1, -1)
        yhat = model.predict(x_input)[0]
        preds.append(yhat)
        last_lags.append(yhat)

    future_index = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=horizon,
                                 freq="D")
    forecast_series = pd.Series(preds, index=future_index, name="forecast")
    return forecast_series


# ---------- MAIN PAGE ----------
st.subheader(f"Center: {selected_center} | Pollutant: {selected_pollutant.upper()}")

# Historical plot
st.markdown("### Historical Trend")

hist_fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_center["date"], df_center[selected_pollutant], label="Historical")
ax.set_xlabel("Date")
ax.set_ylabel(selected_pollutant.upper())
ax.set_title(f"Historical {selected_pollutant.upper()} at {selected_center}")
ax.legend()
st.pyplot(hist_fig)

# Forecast
st.markdown("### Forecast")

try:
    series = df_center.set_index("date")[selected_pollutant]
    forecast_series = make_forecast(series, horizon=forecast_horizon, n_lags=N_LAGS)

    tail_days = max(60, forecast_horizon * 2)
    combined_hist = series.tail(tail_days)

    fc_fig, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(combined_hist.index, combined_hist.values, label="Historical")
    ax2.plot(forecast_series.index, forecast_series.values, "--", label="Forecast")
    ax2.set_xlabel("Date")
    ax2.set_ylabel(selected_pollutant.upper())
    ax2.set_title(f"{selected_pollutant.upper()} Forecast for next {forecast_horizon} days\n({selected_center})")
    ax2.legend()
    st.pyplot(fc_fig)

    st.write("**Forecast values:**")
    st.dataframe(forecast_series.round(2).to_frame())

except Exception as e:
    st.error("Error during forecasting:")
    st.code(str(e))

# Summary stats
st.markdown("### Summary statistics (selected center + date range)")
st.write(df_center[selected_pollutant].describe().round(2))

# src/preprocess.py
import pandas as pd

def make_features(df, target="pm2_5", lags=[1,3,6,12,24]):
    """
    Minimal feature creation for training:
    - ensures timestamp, sorts
    - creates hour, dow, and simple lag features for the target
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # time features
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek

    # lag features for target (if target missing in some rows, result will have NaNs)
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)

    # drop rows that don't have the basic lag features
    required = [f"{target}_lag_{lag}" for lag in lags]
    df = df.dropna(subset=required).reset_index(drop=True)

    return df


def add_target(df, target="pm2_5", horizon=24):
    """
    Adds a future target column named '<target>_future' shifted by -horizon.
    Drops rows without the future target.
    """
    df = df.copy()
    df[f"{target}_future"] = df[target].shift(-horizon)
    df = df.dropna(subset=[f"{target}_future"]).reset_index(drop=True)
    return df
def make_realtime_features(rt_row, history_df, target="pm2_5"):
    """
    Build real-time features using:
    - current realtime row
    - last 24 rows of history_df
    Returns a row with same feature columns as training.
    """
    import pandas as pd

    rt = rt_row.copy()
    hist = history_df.copy()

    # Convert to datetime
    rt["timestamp"] = pd.to_datetime(rt["timestamp"], errors="coerce")
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")

    # Force tz-naive timestamps
    try:
        rt["timestamp"] = rt["timestamp"].dt.tz_localize(None)
    except:
        pass

    try:
        hist["timestamp"] = hist["timestamp"].dt.tz_localize(None)
    except:
        pass

    # Combine history + realtime
    full = pd.concat([hist, rt], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    # Time features
    full["hour"] = full["timestamp"].dt.hour
    full["dow"] = full["timestamp"].dt.dayofweek

    # Lag features
    for lag in [1, 3, 6, 12, 24]:
        full[f"{target}_lag_{lag}"] = full[target].shift(lag)

    # Final row = realtime row
    X = full.tail(1)

    FEATURES = [
        "pm2_5", "pm10", "o3", "no2", "so2", "co",
        "hour", "dow",
        "pm2_5_lag_1", "pm2_5_lag_3", "pm2_5_lag_6",
        "pm2_5_lag_12", "pm2_5_lag_24"
    ]

    # Only keep features present
    X = X[[f for f in FEATURES if f in X.columns]]

    return X

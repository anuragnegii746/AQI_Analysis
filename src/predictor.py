import pandas as pd
from src.realtime import get_realtime_aqi_by_coords, format_for_model
from src.preprocess import make_realtime_features
from src.model import predict_pm2_5
from src.centers import CENTERS


def predict_for_center(center_name):
    """
    Predict PM2.5 for a given AQI center.
    Uses:
    - Center coordinates from centers.py
    - Realtime API data
    - Center-specific history file
    """

    if center_name not in CENTERS:
        raise ValueError(f"Center '{center_name}' not found in CENTERS")

    coords = CENTERS[center_name]

    # 1. Fetch realtime data
    rt = format_for_model(
        get_realtime_aqi_by_coords(coords["lat"], coords["lon"])
    )

    # 2. Load that center's history
    history_path = f"data/history/{center_name}.csv"
    hist = pd.read_csv(history_path, parse_dates=["timestamp"])

    # 3. Build ML features
    X = make_realtime_features(rt, hist)

    # 4. Predict
    pred = predict_pm2_5(X)

    return pred

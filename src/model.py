# src/model.py
import joblib
import os
import pandas as pd

# -------- LOAD MODEL & SCALER --------
def load_model():
    model_path = os.path.join("src", "models", "model.pkl")
    scaler_path = os.path.join("src", "models", "scaler.pkl")

    model = joblib.load(model_path)

    # scaler is optional
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = None

    return model, scaler


# -------- PREDICT FUNCTION --------
def predict_pm2_5(X):
    """
    X = DataFrame returned from make_realtime_features()
    returns predicted pm2_5 (future value)
    """
    model, scaler = load_model()

    # apply scaling if scaler exists
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    pred = model.predict(X_scaled)
    return float(pred[0])
# append line to data/predictions_log.csv
import pandas as pd, os, datetime
def log_prediction(pred, features_df):
    out = features_df.copy()
    out["pred_pm2_5"] = pred
    out["pred_time"] = pd.Timestamp.now()
    os.makedirs("data", exist_ok=True)
    mode = "a" if os.path.exists("data/predictions_log.csv") else "w"
    header = not os.path.exists("data/predictions_log.csv")
    out.to_csv("data/predictions_log.csv", index=False, mode=mode, header=header)

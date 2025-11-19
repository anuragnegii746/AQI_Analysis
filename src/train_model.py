# src/train_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

from src.preprocess import make_features, add_target


def load_data():
    path = os.path.join("data", "processed_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected training file not found: {path}")
    return pd.read_csv(path)


def train_model():
    print("➡ Loading data...")
    df = load_data()

    print("➡ Creating features...")
    df = make_features(df, target="pm2_5")
    df = add_target(df, target="pm2_5")

    # drop obvious non-feature columns
    df = df.copy()
    drop_cols = ["timestamp", "station_id", "center"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Identify numeric feature columns only
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Ensure we don't include the target in X
    target_col = "pm2_5_future"
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe. Columns: {df.columns.tolist()}")
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if not numeric_cols:
        print("ERROR: No numeric feature columns found for training. Current columns:")
        print(df.columns.tolist())
        return

    X = df[numeric_cols]
    y = df[target_col]

    print("Feature columns used for training:", numeric_cols)

    # train/test split (time-aware: no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train model
    print("➡ Training RandomForest...")
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    score = model.score(X_test_scaled, y_test)
    print(f"✅ Model trained. R2 score = {score:.3f}")

    # Save
    os.makedirs("src/models", exist_ok=True)
    joblib.dump(model, "src/models/model.pkl")
    joblib.dump(scaler, "src/models/scaler.pkl")

    print("🎉 Model saved successfully!")
    print("   → src/models/model.pkl")
    print("   → src/models/scaler.pkl")


if __name__ == "__main__":
    train_model()

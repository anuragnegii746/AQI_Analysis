# src/create_history.py
import os
import pandas as pd
from src.centers import CENTERS

IN = os.path.join("data", "processed_data.csv")
OUT_DIR = os.path.join("data", "history")
N = 24  # last N rows (hours)

def normalize_text(s):
    return str(s).strip().lower()

def find_rows_for_center(df, center_key):
    """
    Fuzzy match: look for center_key words inside the 'center' column (case-insensitive).
    Example: center_key 'anand_vihar' matches 'Anand Vihar Delhi - Copy'.
    """
    key = center_key.replace("_", " ").lower()
    if "center" not in df.columns:
        return pd.DataFrame()  # no column to match on
    mask = df["center"].astype(str).str.lower().str.contains(key)
    return df[mask]

def main():
    if not os.path.exists(IN):
        raise FileNotFoundError(f"Input file not found: {IN}")

    df = pd.read_csv(IN)
    # ensure timestamp is datetime and sorted
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    else:
        print("Warning: 'timestamp' column not found - cannot create history files.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    created = []
    skipped = []

    for center_key in CENTERS.keys():
        rows = find_rows_for_center(df, center_key)
        if rows.empty:
            skipped.append(center_key)
            continue

        # keep only timestamp + pm2_5 + other pollutants if present
        keep_cols = ["timestamp", "pm2_5"]
        for c in ["pm10","o3","no2","so2","co"]:
            if c in rows.columns:
                keep_cols.append(c)
        rows = rows[keep_cols].dropna(subset=["timestamp"])

        # take last N rows
        last = rows.tail(N)
        if last.empty:
            skipped.append(center_key)
            continue

        out_path = os.path.join(OUT_DIR, f"{center_key}.csv")
        last.to_csv(out_path, index=False)
        created.append((center_key, len(last), out_path))

    print("History creation summary:")
    if created:
        for c, cnt, path in created:
            print(f"  CREATED {cnt} rows -> {path}")
    if skipped:
        print("\nSkipped (no matching rows found):")
        for c in skipped:
            print("  ", c)
    if not created and not skipped:
        print("No centers processed (check input file and src/centers.py).")

if __name__ == "__main__":
    main()

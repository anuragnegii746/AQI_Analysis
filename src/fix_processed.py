# src/fix_processed.py
import pandas as pd
import os
import numpy as np

IN = os.path.join("data", "processed_data.csv")  # prefer this if present
FALLBACK = os.path.join("data", "processed", "merged_output_final.csv")
OUT = os.path.join("data", "processed_data.csv")

TS_CANDS = ["timestamp","datetime","date_time","date","time","created_at"]
PM25_CANDS = ["pm2_5","pm2.5","pm25","pm_2_5","pm_25","pm"]

def normalize_colname(c):
    return c.strip().lower().replace(" ", "_")

def find_column(cols, candidates):
    cols_map = {normalize_colname(c): c for c in cols}
    for cand in candidates:
        key = normalize_colname(cand)
        if key in cols_map:
            return cols_map[key]
    # fuzzy contains
    for orig in cols:
        low = normalize_colname(orig)
        for cand in candidates:
            if normalize_colname(cand).replace("_","") in low.replace("_",""):
                return orig
    return None

def load_input():
    if os.path.exists(IN):
        print("Loading existing:", IN)
        return pd.read_csv(IN)
    if os.path.exists(FALLBACK):
        print("Loading fallback merged file:", FALLBACK)
        return pd.read_csv(FALLBACK)
    raise FileNotFoundError("Neither data/processed_data.csv nor data/processed/merged_output_final.csv found.")

def coerce_numeric_series(s):
    # strip spaces, replace empty strings with NaN, convert to numeric
    s = s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
    return pd.to_numeric(s, errors="coerce")

def main():
    df = load_input()
    print("Original columns:", df.columns.tolist())

    # Trim column names and build a mapping to original names
    orig_cols = list(df.columns)
    normalized = {orig: normalize_colname(orig) for orig in orig_cols}
    # rename to normalized friendly names temporarily
    df = df.rename(columns={orig: normalized[orig] for orig in orig_cols})

    # Detect timestamp and pm2_5 columns (by normalized names)
    ts_col = find_column(df.columns, TS_CANDS)
    pm_col = find_column(df.columns, PM25_CANDS)

    if ts_col:
        print("Detected timestamp column (normalized):", ts_col)
    else:
        print("No obvious timestamp column found. Columns:", df.columns.tolist())
        return

    if pm_col:
        print("Detected pm2.5 column (normalized):", pm_col)
    else:
        print("No obvious PM2.5 column found; will try first numeric column.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            pm_col = numeric_cols[0]
            print("Using numeric column:", pm_col)
        else:
            # try to coerce any column later; we'll inspect
            print("No numeric columns detected. Columns are:", df.columns.tolist())
            return

    # Strip whitespace from object/string columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip().replace({"": np.nan})

    # Parse timestamp column
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", infer_datetime_format=True)
    n_bad_ts = df[ts_col].isna().sum()
    print(f"Timestamps parsed: {len(df)-n_bad_ts} valid, {n_bad_ts} invalid.")

    # Coerce pm2_5 to numeric
    df[pm_col] = coerce_numeric_series(df[pm_col])
    n_bad_pm = df[pm_col].isna().sum()
    print(f"PM2.5 conversion: {len(df)-n_bad_pm} numeric, {n_bad_pm} non-numeric/NaN.")

    # Drop rows missing timestamp or pm2_5
    before = len(df)
    df = df.dropna(subset=[ts_col, pm_col]).reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before-after} rows without timestamp or pm2_5. Remaining rows: {after}")

    # Rename columns to exact names expected by pipeline
    # ensure final names: timestamp, pm2_5 (others left as-is)
    df = df.rename(columns={ts_col: "timestamp", pm_col: "pm2_5"})

    # Optionally: fill small numeric gaps using median for pm2_5 (uncomment if you prefer imputation)
    # df["pm2_5"] = df["pm2_5"].fillna(df["pm2_5"].median())

    # Save cleaned file
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT, index=False)
    print("Saved cleaned file to:", OUT)
    print("Preview:")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()

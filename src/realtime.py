# src/realtime.py
import time
import logging
from typing import Optional, Dict, Any
import requests
import pandas as pd

API_KEY = "d4ae8a9567e2e38f4649e0f3f40cc607"
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def _call_api(params: dict, retries: int = 2, backoff: float = 1.0) -> Optional[dict]:
    params = params.copy()
    params["appid"] = API_KEY
    for attempt in range(1, retries + 2):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning("OpenWeather API call failed (attempt %d): %s", attempt, e)
            if attempt <= retries:
                time.sleep(backoff * attempt)
            else:
                logger.error("OpenWeather API failed after %d attempts.", attempt)
                return None

def _normalize_response(resp: dict) -> Optional[Dict[str, Any]]:
    if not resp or "list" not in resp or not resp["list"]:
        return None
    item = resp["list"][0]
    out = {}
    out["aqi_owm"] = item.get("main", {}).get("aqi")
    comps = item.get("components", {})
    out["pm2_5"] = comps.get("pm2_5")
    out["pm10"] = comps.get("pm10")
    out["so2"] = comps.get("so2")
    out["no2"] = comps.get("no2")
    out["o3"] = comps.get("o3")
    out["co"] = comps.get("co")
    dt = item.get("dt")
    out["timestamp_utc"] = pd.to_datetime(dt, unit="s", utc=True) if dt else pd.Timestamp.utcnow()
    return out

def get_realtime_aqi_by_coords(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    params = {"lat": float(lat), "lon": float(lon)}
    resp = _call_api(params)
    result = _normalize_response(resp)
    if result is None:
        logger.error("Failed to normalize OpenWeather response for coords %s,%s", lat, lon)
    return result

def format_for_model(d: Dict[str, Any], station_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    if d is None:
        return None
    row = {
        "timestamp": d.get("timestamp_utc", pd.Timestamp.utcnow()),
        "station_id": station_id or "realtime",
        "pm2_5": d.get("pm2_5"),
        "pm10": d.get("pm10"),
        "no2": d.get("no2"),
        "so2": d.get("so2"),
        "o3": d.get("o3"),
        "co": d.get("co"),
        "aqi_owm": d.get("aqi_owm"),
    }
    df = pd.DataFrame([row])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

if __name__ == "__main__":
    lat, lon = 28.6139, 77.2090
    d = get_realtime_aqi_by_coords(lat, lon)
    print("Raw normalized dict:\n", d)
    df = format_for_model(d, station_id="delhi_center")
    print("\nDataFrame for model:\n", df.to_dict(orient='records'))

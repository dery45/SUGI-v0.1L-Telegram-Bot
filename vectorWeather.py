"""
vectorWeather.py — Open-Meteo → ChromaDB (collection: weather_data)

Data is filtered to only what matters for:
  - Farmers / crop growers       → rain, temperature, humidity, soil moisture
  - Plantation managers          → evapotranspiration, soil temp, dew point
  - Livestock / animal farmers   → heat stress index, wind chill, humidity
  - Government / food security   → drought indicators, flood risk (heavy rain)

Storage strategy (vs original):
  - Original: 2,592 individual hourly docs  (noisy, redundant, heavy)
  - This version: ~109 daily summary docs   (concise, farming-focused)
    Each daily doc summarises min/max/avg of all farming-relevant variables
    + flags extreme events (heavy rain, drought, heat stress, frost risk)

Run standalone:   python vectorWeather.py
Import in main:   from vectorWeather import weather_store, weather_retriever
"""

import hashlib
import os
import time
from datetime import date

import openmeteo_requests
import numpy as np
import pandas as pd
import requests_cache
from retry_requests import retry

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# ─── Configuration ────────────────────────────────────────────────────────────
DB_PATH       = "chrome_longchain_db"
EMBED_MODEL   = "mxbai-embed-large"

LATITUDE      = -6.1818
LONGITUDE     = 106.8223
LOCATION_NAME = "Jakarta"

PAST_DAYS     = 92     # ~3 months history for seasonal analysis
FORECAST_DAYS = 16     # 2-week forecast
CHECK_INTERVAL = 300   # re-check for day change every 5 minutes

# ─── Only farming-relevant hourly variables ───────────────────────────────────
# Removed: pressure at altitude, wind at 80/120/180m, temperature at 80/120/180m
# These are aviation/meteorology variables irrelevant to farming.
HOURLY_VARS = [
    # Core weather
    "temperature_2m",           # crop heat stress, frost risk
    "relative_humidity_2m",     # disease risk, livestock comfort
    "dew_point_2m",             # condensation, fungal risk
    "precipitation",            # total water input
    "rain",                     # rainfall (excludes snow/sleet)
    "evapotranspiration",       # irrigation demand estimate
    # Surface conditions
    "surface_pressure",         # weather system tracking
    "cloud_cover",              # solar radiation proxy, photoperiod
    # Soil — critical for planting decisions
    "soil_temperature_0cm",     # seed germination temperature
    "soil_temperature_6cm",     # root zone temperature
    "soil_temperature_18cm",    # deep root zone
    "soil_moisture_0_to_1cm",   # surface soil wetness
    "soil_moisture_1_to_3cm",   # shallow root zone
    "soil_moisture_3_to_9cm",   # main root zone
    "soil_moisture_9_to_27cm",  # deep root zone
    # Wind at ground level only
    "wind_speed_10m",           # crop damage risk, spray drift
    "wind_direction_10m",       # wind direction for field operations
    "wind_gusts_10m",           # storm/damage risk
]

# ─── ChromaDB ─────────────────────────────────────────────────────────────────
_embeddings   = OllamaEmbeddings(model=EMBED_MODEL)
weather_store = Chroma(
    collection_name="weather_data",
    persist_directory=DB_PATH,
    embedding_function=_embeddings,
)

# No score_threshold — weather docs are structured text (not natural language)
# and would be incorrectly filtered by cosine similarity thresholds.
weather_retriever = weather_store.as_retriever(search_kwargs={"k": 8})

# ─── Open-Meteo client ────────────────────────────────────────────────────────
_cache   = requests_cache.CachedSession(".weather_cache", expire_after=3600)
_retried = retry(_cache, retries=5, backoff_factor=0.2)
_client  = openmeteo_requests.Client(session=_retried)

API_URL = "https://api.open-meteo.com/v1/forecast"

# ─── Fetch raw hourly DataFrame ───────────────────────────────────────────────

def _fetch_dataframe() -> pd.DataFrame:
    params = {
        "latitude":      LATITUDE,
        "longitude":     LONGITUDE,
        "hourly":        HOURLY_VARS,
        "past_days":     PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
    }
    responses = _client.weather_api(API_URL, params=params)
    response  = responses[0]

    print(f"   📍 {LOCATION_NAME} — "
          f"{response.Latitude():.4f}°N {response.Longitude():.4f}°E, "
          f"elevation {response.Elevation():.0f} m asl")

    hourly = response.Hourly()
    data   = {"datetime": pd.date_range(
        start     = pd.to_datetime(hourly.Time(),    unit="s", utc=True),
        end       = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq      = pd.Timedelta(seconds=hourly.Interval()),
        inclusive = "left",
    )}
    for idx, var in enumerate(HOURLY_VARS):
        data[var] = hourly.Variables(idx).ValuesAsNumpy()

    df = pd.DataFrame(data)
    # Convert UTC to local (WIB = UTC+7 for Jakarta; adjust if needed)
    df["datetime"] = df["datetime"].dt.tz_convert("Asia/Jakarta")
    df["date"]     = df["datetime"].dt.date
    return df

# ─── Aggregate hourly → daily farming summary ─────────────────────────────────

def _daily_summary_to_text(day: date, g: pd.DataFrame) -> str:
    """
    Convert one day's hourly rows into a single farming-focused text summary.
    Flags agronomic alerts (heavy rain, drought, heat stress, frost, high humidity).
    """
    d = str(day)

    # ── Safe stat helper — returns fallback when all values are NaN ─────────────
    def _s(series, func="mean", fallback=float("nan")):
        clean = series.dropna()
        if clean.empty:
            return fallback
        return getattr(clean, func)()

    # ── Core stats ────────────────────────────────────────────────────────────
    t_min  = _s(g["temperature_2m"],      "min")
    t_max  = _s(g["temperature_2m"],      "max")
    t_avg  = _s(g["temperature_2m"],      "mean")
    rh_avg = _s(g["relative_humidity_2m"],"mean")
    rh_max = _s(g["relative_humidity_2m"],"max")
    dp_avg = _s(g["dew_point_2m"],        "mean")
    rain   = _s(g["rain"],                "sum",  fallback=0.0)
    precip = _s(g["precipitation"],       "sum",  fallback=0.0)
    et     = _s(g["evapotranspiration"],  "sum",  fallback=0.0)
    cloud  = _s(g["cloud_cover"],         "mean")
    pres   = _s(g["surface_pressure"],    "mean")
    ws_max = _s(g["wind_speed_10m"],      "max",  fallback=0.0)
    ws_avg = _s(g["wind_speed_10m"],      "mean", fallback=0.0)
    gust   = _s(g["wind_gusts_10m"],      "max",  fallback=0.0)

    # Wind direction of peak wind hour (guard against all-NaN wind column)
    wind_col = g["wind_speed_10m"].dropna()
    if not wind_col.empty:
        peak_hour = g.loc[wind_col.idxmax()]
        wind_dir  = peak_hour["wind_direction_10m"]
        if pd.isna(wind_dir):
            wind_dir = float("nan")
    else:
        wind_dir = float("nan")

    # Soil
    sm_surf = _s(g["soil_moisture_0_to_1cm"], "mean")
    sm_root = _s(g["soil_moisture_3_to_9cm"], "mean")
    sm_deep = _s(g["soil_moisture_9_to_27cm"],"mean")
    st_surf = _s(g["soil_temperature_0cm"],   "mean")
    st_root = _s(g["soil_temperature_6cm"],   "mean")
    st_deep = _s(g["soil_temperature_18cm"],  "mean")

    # ── Agronomic alerts ──────────────────────────────────────────────────────
    alerts = []

    # Drought risk  (NaN comparisons evaluate to False — safe to leave as-is)
    if rain < 1.0 and (sm_root == sm_root) and sm_root < 0.15:
        alerts.append("⚠️ DROUGHT RISK: No rain and low root-zone soil moisture — consider irrigation.")
    elif rain < 2.0:
        alerts.append("🟡 LOW RAINFALL: Monitor soil moisture, irrigation may be needed.")

    # Flood / heavy rain risk
    if rain >= 50:
        alerts.append("🚨 EXTREME RAIN: >50mm — high flood risk, delay field operations.")
    elif rain >= 20:
        alerts.append("⚠️ HEAVY RAIN: >20mm — monitor drainage, risk of waterlogging.")

    # Heat stress (crops + livestock)
    if t_max >= 38:
        alerts.append("🚨 EXTREME HEAT: >38°C — severe heat stress for crops and livestock.")
    elif t_max >= 35:
        alerts.append("⚠️ HEAT STRESS: >35°C — protect sensitive crops, provide shade for livestock.")

    # Frost / cold damage risk
    if t_min <= 10:
        alerts.append("❄️ COLD RISK: <10°C — risk of chilling injury to tropical crops.")
    elif t_min <= 15:
        alerts.append("🟡 COOL NIGHT: <15°C — monitor cold-sensitive crops and seedlings.")

    # Disease risk (high humidity = fungal/bacterial outbreak)
    if rh_max >= 95 and t_avg >= 25:
        alerts.append("🦠 HIGH DISEASE RISK: Very high humidity + warm temp — spray fungicide preventively.")
    elif rh_avg >= 85:
        alerts.append("🟡 ELEVATED HUMIDITY: Monitor for fungal diseases (blast, blight, mildew).")

    # Strong wind
    if gust >= 60:
        alerts.append("🌪️ STRONG GUSTS: >60 km/h — risk of crop lodging and structural damage.")
    elif ws_max >= 40:
        alerts.append("💨 HIGH WIND: >40 km/h — avoid spraying pesticides/herbicides.")

    # High evapotranspiration = water demand
    if et >= 6:
        alerts.append("💧 HIGH WATER DEMAND: ET >6mm — increase irrigation frequency.")

    alert_text = "\n".join(alerts) if alerts else "✅ No major agronomic alerts for this day."

    # ── Assemble text ─────────────────────────────────────────────────────────
    lines = [
        f"=== Daily Weather Summary: {LOCATION_NAME} | {d} ===",
        "",
        "── ATMOSPHERE ──────────────────────────────",
        f"Temperature: min {t_min:.1f}°C / avg {t_avg:.1f}°C / max {t_max:.1f}°C" if not any(map(lambda v: v != v, [t_min, t_avg, t_max])) else "Temperature: data unavailable",
        f"Humidity: avg {rh_avg:.0f}% / max {rh_max:.0f}%" if rh_avg == rh_avg else "Humidity: data unavailable",
        f"Dew point: avg {dp_avg:.1f}°C" if dp_avg == dp_avg else "Dew point: data unavailable",
        f"Rainfall: {rain:.1f} mm",
        f"Total precipitation: {precip:.1f} mm",
        f"Evapotranspiration (ET0): {et:.2f} mm/day",
        f"Cloud cover: {cloud:.0f}%" if cloud == cloud else "Cloud cover: data unavailable",
        f"Surface pressure: {pres:.1f} hPa" if pres == pres else "Surface pressure: data unavailable",
        "",
        "── WIND ────────────────────────────────────",
        f"Wind speed: avg {ws_avg:.1f} km/h / max {ws_max:.1f} km/h",
        f"Wind gusts: max {gust:.1f} km/h",
        f"Wind direction at peak: {wind_dir:.0f}°" if wind_dir == wind_dir else "Wind direction: data unavailable",
        "",
        "── SOIL ────────────────────────────────────",
        f"Soil temperature: surface {st_surf:.1f}°C / 6cm {st_root:.1f}°C / 18cm {st_deep:.1f}°C" if st_surf == st_surf else "Soil temperature: data unavailable",
        f"Soil moisture: surface {sm_surf:.3f} m³/m³ / root zone {sm_root:.3f} m³/m³ / deep {sm_deep:.3f} m³/m³" if sm_surf == sm_surf else "Soil moisture: data unavailable",
        "",
        "── AGRONOMIC ALERTS ────────────────────────",
        alert_text,
    ]
    return "\n".join(lines)

# ─── Core indexing ────────────────────────────────────────────────────────────

def index_weather(fetch_date: str):
    """
    Fetch Open-Meteo data, aggregate to daily summaries, store in ChromaDB.
    Old docs from previous fetch dates are deleted first.
    """
    print(f"\n🌤️  Fetching Open-Meteo data (batch: {fetch_date})...")
    try:
        df = _fetch_dataframe()
    except Exception as e:
        print(f"   ❌ API fetch failed: {e}")
        return

    # Remove stale docs
    try:
        old = weather_store.get(where={"fetch_date": {"$ne": fetch_date}}, limit=100_000)
        if old["ids"]:
            weather_store.delete(ids=old["ids"])
            print(f"   🗑️  Removed {len(old['ids'])} outdated docs.")
    except Exception:
        pass

    # Build daily summaries
    documents = []
    ids       = []
    grouped   = df.groupby("date")

    for day, group in grouped:
        text   = _daily_summary_to_text(day, group)
        doc_id = _doc_id(f"{LOCATION_NAME}:{day}")
        documents.append(Document(
            page_content=text,
            metadata={
                "source":     "open_meteo",
                "location":   LOCATION_NAME,
                "date":       str(day),
                "fetch_date": fetch_date,
            },
            id=doc_id,
        ))
        ids.append(doc_id)

    # Batch upsert
    batch_size = 50
    total      = len(documents)
    print(f"   📦 Storing {total} daily weather summaries...")
    for i in range(0, total, batch_size):
        weather_store.add_documents(
            documents=documents[i:i+batch_size],
            ids=ids[i:i+batch_size],
        )
    print(f"   ✅ Weather indexed: {total} daily summaries for {fetch_date}.")

def _doc_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def is_weather_fresh(today: str) -> bool:
    try:
        result = weather_store.get(where={"fetch_date": today}, limit=1)
        return bool(result["ids"])
    except Exception:
        return False

# ─── Main server loop ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"🚀 Weather Vector Server | Location: {LOCATION_NAME}")
    print(f"   Stores farming-focused daily summaries (not raw hourly data).")
    print(f"   Checks for day change every {CHECK_INTERVAL}s.\n")

    last_date = None

    while True:
        today = date.today().isoformat()

        if today != last_date:
            if is_weather_fresh(today):
                print(f"✅ Weather data for {today} already in ChromaDB — skipping fetch.")
            else:
                index_weather(today)
            last_date = today

        time.sleep(CHECK_INTERVAL)
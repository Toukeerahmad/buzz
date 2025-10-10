from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib
import folium

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "buzz-complete.csv"
MODEL_PATH = BASE_DIR / "buzz_model.pkl"
PREVIEW_CSV = BASE_DIR / "preview_top.csv"
PREVIEW_HTML = BASE_DIR / "preview_map.html"


def extract_first_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value)
    matches = re.findall(r"[-+]?[0-9]*\.?[0-9]+", s)
    if not matches:
        return np.nan
    try:
        return float(matches[0])
    except Exception:
        return np.nan


def tokenize_count(value: object) -> int:
    if pd.isna(value):
        return 0
    tokens = re.split(r"[,;|/]", str(value))
    tokens = [t.strip() for t in tokens if t.strip()]
    return len(tokens)


def prepare_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["metro_binary"] = (
        df.get("MetroAccess", "No").astype(str).str.strip().str.lower().str.startswith("y").astype(int)
    )
    df["demand_token_count"] = df.get("DemandProducts", "").apply(tokenize_count).astype(int)

    if "CrimeReportedCitywide" in df.columns:
        df["CrimeReportedCitywide"] = df["CrimeReportedCitywide"].apply(extract_first_number)

    for col in [
        "Cost_of_Living",
        "Population",
        "Avg_Income",
        "Competition_Index",
        "Footfall_Potential",
        "AvgRent2BHK",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for c in feature_columns:
        if c not in df.columns:
            df[c] = np.nan

    return df[feature_columns].copy()


def show_map(df: pd.DataFrame, business_label: str) -> None:
    m = folium.Map(location=[12.97, 77.59], zoom_start=5)
    for _, row in df.iterrows():
        area = str(row.get("Area", "Unknown"))
        district = str(row.get("District", "Unknown"))
        lat = 12.97 + (hash(area) % 10) * 0.01
        lon = 77.59 + (hash(district) % 10) * 0.01
        score = float(row.get("predicted_success_ratio", 0.0))
        popup = f"{area}<br>Success: {score:.2f}<br>Business: {business_label}"
        folium.Marker([lat, lon], popup=popup).add_to(m)
    m.save(PREVIEW_HTML)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    df = pd.read_csv(DATA_PATH)
    X = prepare_features(df, feature_columns)
    preds = pipeline.predict(X)

    df = df.copy()
    df["predicted_success_ratio"] = preds
    df_sorted = df.sort_values("predicted_success_ratio", ascending=False)
    top_df = df_sorted.head(5).copy()

    cols = [
        "District",
        "Taluk",
        "Area",
        "predicted_success_ratio",
        "Footfall_Potential",
        "Competition_Index",
    ]
    for c in cols:
        if c not in top_df.columns:
            top_df[c] = np.nan

    top_df.to_csv(PREVIEW_CSV, index=False)
    show_map(top_df, business_label="Preview")

    print("Preview written:")
    print(f" - CSV: {PREVIEW_CSV}")
    print(f" - Map: {PREVIEW_HTML}")
    print("\nTop 5 (District | Taluk | Area | Predicted | Footfall | Competition):")
    print(top_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

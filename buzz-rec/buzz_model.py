"""
BuzzLocator model builder

This script:
- Loads buzz-complete.csv in the same folder
- Engineers features per spec (metro_binary, demand_token_count)
- Computes weighted success_score and normalizes to [0, 1]
- Trains a MinMaxScaler + RandomForestRegressor on selected features
- Saves the trained model to buzz_model.pkl using joblib
- Writes a processed CSV with the success_score column
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "buzz-complete.csv"
MODEL_PATH = BASE_DIR / "buzz_model.pkl"
PROCESSED_PATH = BASE_DIR / "buzz-processed.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Expected dataset at {DATA_PATH}. Place 'buzz-complete.csv' in this folder.")


# -------------------------
# Helpers
# -------------------------
def extract_first_number(value: object) -> float:
    """Extract first numeric token from a value; return NaN if none."""
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
    """Count comma/semicolon/pipe-separated tokens after trimming."""
    if pd.isna(value):
        return 0
    tokens = re.split(r"[,;|/]", str(value))
    tokens = [t.strip() for t in tokens if t.strip()]
    return len(tokens)


# -------------------------
# Load data
# -------------------------
df = pd.read_csv(DATA_PATH)


# -------------------------
# Feature engineering (metro_binary, demand_token_count)
# -------------------------
df["metro_binary"] = (
    df.get("MetroAccess", "No").astype(str).str.strip().str.lower().str.startswith("y").astype(int)
)
df["demand_token_count"] = df.get("DemandProducts", "").apply(tokenize_count).astype(int)


# -------------------------
# Compute success_score per weighted formula
# success_score = (
#     0.35 * Footfall_Potential +
#     0.20 * Avg_Income +
#     0.20 * (1 - Competition_Index) +
#     0.10 * (1 - CrimeReportedCitywide) +
#     0.10 * demand_token_count +
#     0.05 * metro_binary
# )
# with all terms normalized to [0, 1]
# -------------------------

formula_cols: List[str] = [
    "Footfall_Potential",
    "Avg_Income",
    "Competition_Index",
    "CrimeReportedCitywide",
]

# Coerce numerics
for col in [
    "Cost_of_Living",
    "Population",
    "Avg_Income",
    "Competition_Index",
    "Footfall_Potential",
    "AvgRent2BHK",
    "CrimeReportedCitywide",
]:
    if col in df.columns:
        if col == "CrimeReportedCitywide":
            df[col] = df[col].apply(extract_first_number)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

# MinMax scale components used in the formula
minmax_scaler = MinMaxScaler()
scaled_formula = {}
for col in formula_cols:
    if col not in df.columns:
        df[col] = np.nan
    values = df[[col]].copy()
    # Impute median for scaling stability
    median_imputer = SimpleImputer(strategy="median")
    values_imputed = median_imputer.fit_transform(values)
    scaled = minmax_scaler.fit_transform(values_imputed)
    scaled_formula[col] = scaled.reshape(-1)

footfall_norm = scaled_formula["Footfall_Potential"]
income_norm = scaled_formula["Avg_Income"]
competition_norm = scaled_formula["Competition_Index"]
crime_norm = scaled_formula["CrimeReportedCitywide"]

# demand_token_count is already numeric; scale to [0,1]
demand_scaled = MinMaxScaler().fit_transform(df[["demand_token_count"]].fillna(0)).reshape(-1)
metro_binary = df["metro_binary"].fillna(0).astype(int).values

success_score_raw = (
    0.35 * footfall_norm
    + 0.20 * income_norm
    + 0.20 * (1.0 - competition_norm)
    + 0.10 * (1.0 - crime_norm)
    + 0.10 * demand_scaled
    + 0.05 * metro_binary
)

# Normalize success_score to [0, 1]
success_score = MinMaxScaler().fit_transform(success_score_raw.reshape(-1, 1)).reshape(-1)
df["success_score"] = success_score


# -------------------------
# Train RandomForest on specified features
# -------------------------
feature_columns: List[str] = [
    "Cost_of_Living",
    "Population",
    "Avg_Income",
    "Competition_Index",
    "Footfall_Potential",
    "AvgRent2BHK",
    "CrimeReportedCitywide",
    "metro_binary",
    "demand_token_count",
]

# Ensure all features exist and are numeric
for col in feature_columns:
    if col not in df.columns:
        df[col] = np.nan
    df[col] = pd.to_numeric(df[col], errors="coerce")

X = df[feature_columns].copy()
y = df["success_score"].values

pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
            ),
        ),
    ]
)

pipeline.fit(X, y)

# Save model with joblib
joblib.dump({
    "pipeline": pipeline,
    "feature_columns": feature_columns,
}, MODEL_PATH)

# Save processed data
df.to_csv(PROCESSED_PATH, index=False)

print("BuzzLocator model trained successfully.")
print(f" - Model saved to: {MODEL_PATH}")
print(f" - Processed data with success_score saved to: {PROCESSED_PATH}")

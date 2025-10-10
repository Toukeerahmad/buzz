from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import folium
from streamlit.components.v1 import html


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "buzz-complete.csv"
MODEL_PATH = BASE_DIR / "buzz_model.pkl"


# -------------------------
# Helpers
# -------------------------

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


def get_business_types(df: pd.DataFrame) -> List[str]:
    types = set()
    if "DemandProducts" in df.columns:
        for val in df["DemandProducts"].dropna():
            for token in str(val).split(","):
                t = token.strip()
                if t:
                    types.add(t)
    return sorted(types)


def prepare_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    # Recreate engineered features to be safe
    df["metro_binary"] = (
        df.get("MetroAccess", "No").astype(str).str.strip().str.lower().str.startswith("y").astype(int)
    )
    df["demand_token_count"] = df.get("DemandProducts", "").apply(tokenize_count).astype(int)

    # Coerce numerics
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

    # Ensure all required columns exist
    for c in feature_columns:
        if c not in df.columns:
            df[c] = np.nan

    X = df[feature_columns].copy()
    return X


def filter_by_business_type(df: pd.DataFrame, business_type: str | None) -> pd.DataFrame:
    if not business_type:
        return df
    if "DemandProducts" not in df.columns:
        return df
    mask = df["DemandProducts"].astype(str).str.contains(fr"\b{re.escape(business_type)}\b", case=False, na=False)
    return df[mask] if mask.any() else df


def show_map(df: pd.DataFrame, business_label: str) -> None:
    m = folium.Map(location=[12.97, 77.59], zoom_start=5)  # Centered on India
    for _, row in df.iterrows():
        area = str(row.get("Area", "Unknown"))
        district = str(row.get("District", "Unknown"))
        # If no coordinates available, generate approximate fake ones using hash()
        lat = 12.97 + (hash(area) % 10) * 0.01
        lon = 77.59 + (hash(district) % 10) * 0.01
        score = float(row.get("predicted_success_ratio", 0.0))
        popup = f"{area}<br>Success: {score:.2f}<br>Business: {business_label}"
        folium.Marker([lat, lon], popup=popup).add_to(m)
    html(folium.Figure().add_child(m).render(), height=420)


# -------------------------
# Cached loaders
# -------------------------

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model() -> Tuple[object, List[str]]:
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]
    return pipeline, feature_columns


# -------------------------
# UI
# -------------------------

st.set_page_config(page_title="BuzzLocator — AI Business Location Recommender", layout="wide")

st.title("BuzzLocator — AI Business Location Recommender")
st.write(
    "Discover the most promising locations to launch your business. "
    "Select a business type and number of locations to see ranked recommendations with an interactive bar chart and map."
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found at {DATA_PATH}. Place 'buzz-complete.csv' in the same folder.")
    st.stop()

try:
    pipeline, feature_columns = load_model()
except Exception as e:
    st.error("Model not found or failed to load. Train it first by running: 'python buzz_model.py'")
    st.exception(e)
    st.stop()

raw_df = load_data()

biz_types = get_business_types(raw_df)
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    selected_biz = st.selectbox("Choose business type", options=biz_types if biz_types else ["General"], index=0)
with col2:
    custom_biz = st.text_input("Or enter a custom business type", value="")
with col3:
    top_n = st.slider("Top N", min_value=5, max_value=30, value=5, step=1)

business_label = custom_biz.strip() or selected_biz

recommend_clicked = st.button("Recommend")

# Always show default top N for quick demo; recompute on click with chosen filters
active_df = filter_by_business_type(raw_df, business_label if recommend_clicked else None)
X = prepare_features(active_df.copy(), feature_columns)
preds = pipeline.predict(X)
active_df = active_df.copy()
active_df["predicted_success_ratio"] = preds
active_df_sorted = active_df.sort_values("predicted_success_ratio", ascending=False)

top_df = active_df_sorted.head(top_n)

# Table
st.subheader(f"Top {top_n} Locations for '{business_label if recommend_clicked else 'All'}'")
columns_to_show = [
    "District",
    "Taluk",
    "Area",
    "predicted_success_ratio",
    "Footfall_Potential",
    "Competition_Index",
]
for c in columns_to_show:
    if c not in top_df.columns:
        top_df[c] = np.nan

display_df = top_df[columns_to_show].rename(columns={"predicted_success_ratio": "Predicted Success Ratio"})
st.dataframe(display_df, use_container_width=True)

# Bar chart
chart_df = top_df[["Area", "predicted_success_ratio"]].set_index("Area")
st.bar_chart(chart_df)

# Map
st.subheader("Map of Top Locations")
show_map(top_df, business_label if recommend_clicked else "")

st.caption("Note: Map markers use approximate coordinates when exact lat/lon is unavailable.")

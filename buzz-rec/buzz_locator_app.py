# buzz_locator_app.py
"""
Streamlit app for BuzzLocator
Automatically trains model if buzz_model.pkl is missing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import dill
import subprocess
import sys

BASE_DIR = Path(__file__).parent
MODEL_BUNDLE = BASE_DIR / "buzz_model.pkl"
CSV_PATH = BASE_DIR / "buzz-complete.csv"

st.set_page_config(page_title="BuzzLocator", layout="wide")
st.title("BuzzLocator — Business Location Recommender")

# -------------------------
# Auto-train if model missing
# -------------------------
if not MODEL_BUNDLE.exists():
    st.warning("Model bundle not found. Training model now (this may take a few minutes)...")
    try:
        subprocess.check_call([sys.executable, str(BASE_DIR / "buzz_model.py")])
        st.success("Model training complete!")
    except Exception as e:
        st.error(f"Failed to train model: {e}")
        st.stop()

# -------------------------
# Load model bundle
# -------------------------
with open(MODEL_BUNDLE, "rb") as f:
    bundle = dill.load(f)

model_name = bundle['model_name']
model = bundle['model']
preprocessor = bundle['preprocessor']
FEATURE_COLS = bundle['feature_cols']
area_col = bundle.get('area_col', None)
business_col = bundle.get('business_col', None)

st.write(f"Using model: **{model_name}**")

# -------------------------
# Load dataset
# -------------------------
if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
else:
    df = pd.DataFrame(columns=FEATURE_COLS)

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("User inputs")
preferred_business = st.sidebar.text_input("Preferred business (optional)", "")
preferred_area = st.sidebar.text_input("Preferred area (optional)", "")
multiselect_areas = []

if area_col and area_col in df.columns:
    all_areas = sorted(df[area_col].dropna().astype(str).unique().tolist())
    multiselect_areas = st.sidebar.multiselect("Compare areas", options=all_areas, default=all_areas[:5])

top_k = st.sidebar.slider("Top areas to show", 1, 20, 5)

# -------------------------
# Prepare rows for prediction
# -------------------------
def make_feature_rows_for_areas(df, areas):
    rows = []
    for a in areas:
        subset = df[df[area_col].astype(str) == str(a)] if area_col in df.columns else pd.DataFrame()
        row = {}
        for c in FEATURE_COLS:
            if c in subset.columns:
                if pd.api.types.is_numeric_dtype(subset[c]):
                    row[c] = subset[c].mean()
                else:
                    row[c] = subset[c].mode().iloc[0] if not subset[c].mode().empty else subset[c].iloc[0]
            else:
                row[c] = np.nan
        if business_col and preferred_business:
            row[business_col] = preferred_business
        if area_col:
            row[area_col] = a
        rows.append(row)
    return pd.DataFrame(rows)

# -------------------------
# Build prediction dataframe
# -------------------------
if multiselect_areas:
    prediction_df = make_feature_rows_for_areas(df, multiselect_areas)
    labels = multiselect_areas
elif preferred_area and area_col and area_col in df.columns:
    prediction_df = make_feature_rows_for_areas(df, [preferred_area])
    labels = [preferred_area]
else:
    labels = df[area_col].dropna().astype(str).unique().tolist()[:top_k] if area_col in df.columns else [f"row-{i}" for i in range(top_k)]
    prediction_df = make_feature_rows_for_areas(df, labels)

# -------------------------
# Predict
# -------------------------
X_prepared = preprocessor.transform(prediction_df[FEATURE_COLS])
preds = model.predict(X_prepared)
preds = np.clip(preds, 0, 100)

result_df = pd.DataFrame({
    'area': labels,
    'predicted_survival_score': np.round(preds, 2)
}).sort_values('predicted_survival_score', ascending=False)

# -------------------------
# Show results
# -------------------------
st.subheader("Predicted survival scores")
st.write("Higher score → better chance of business surviving there")
st.dataframe(result_df.reset_index(drop=True))

# -------------------------
# Comparison chart
# -------------------------
st.subheader("Comparison chart")
fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(result_df))))
ax.barh(result_df['area'][::-1], result_df['predicted_survival_score'][::-1])
ax.set_xlabel("Predicted survival score (0-100)")
ax.set_title("Area comparison")
plt.tight_layout()
st.pyplot(fig)

# -------------------------
# Pros & Cons heuristics
# -------------------------
st.subheader("Pros & Cons (simple heuristic)")
proscons = []
medians = {c: df[c].median() for c in FEATURE_COLS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])}

for i, row in result_df.iterrows():
    area_label = row['area']
    idx = labels.index(area_label) if area_label in labels else None
    feat_row = prediction_df.iloc[idx] if idx is not None else pd.Series({c: np.nan for c in FEATURE_COLS})
    pros, cons = [], []
    for c in FEATURE_COLS:
        if c not in feat_row.index or pd.isna(feat_row[c]):
            continue
        val = float(feat_row[c])
        med = medians.get(c)
        if med is None:
            continue
        lc = c.lower()
        if any(k in lc for k in ['demand','foot','popul','income','sale','rating']):
            if val >= med:
                pros.append(f"{c} above median")
        if any(k in lc for k in ['comp','compet','rent','crime','cost']):
            if val >= med:
                cons.append(f"{c} high (>= median)")
    if preferred_business and business_col:
        pros.append("Matches preferred business")
    if not pros: pros = ["No strong pros detected"]
    if not cons: cons = ["No strong cons detected"]
    proscons.append({'area': area_label, 'pros': "; ".join(pros), 'cons': "; ".join(cons)})

pc_df = pd.DataFrame(proscons).sort_values('area')
st.dataframe(pc_df)

# -------------------------
# Top suggestions
# -------------------------
st.subheader("Top suggestions")
st.write(f"Top {min(len(result_df), top_k)} areas by predicted survival score:")
st.table(result_df.head(top_k).reset_index(drop=True))

st.markdown("""
**Method notes:**  
- Model trained automatically if missing.  
- Uses Linear Regression & Random Forest.  
- Aggregates area-level features and predicts a survival score (0–100).  
- Quality depends on available CSV features (population, demand, footfall, rent, crime, etc.).
""")

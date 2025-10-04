# buzz_locator_app.py
"""
Streamlit app to query the BuzzLocator model.
Ready for Streamlit Cloud deployment.

Ensure the following files are committed in the same folder as this app:
  - buzz_model.pkl
  - buzz-complete.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Paths (robust relative paths)
# -------------------------
BASE_DIR = Path(__file__).parent
MODEL_BUNDLE = BASE_DIR / "buzz_model.pkl"
CSV_PATH = BASE_DIR / "buzz-complete.csv"

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="BuzzLocator", layout="wide")
st.title("BuzzLocator — Business Location Recommender")

# -------------------------
# Check for model existence
# -------------------------
if not MODEL_BUNDLE.exists():
    st.error(
        f"🚨 Model file not found at `{MODEL_BUNDLE}`.\n"
        "Please generate `buzz_model.pkl` locally by running `buzz_model.py` and commit it to the repo."
    )
    st.stop()

# -------------------------
# Load model bundle
# -------------------------
try:
    bundle = joblib.load(MODEL_BUNDLE)
except Exception as e:
    st.error(f"Failed to load model bundle: {e}")
    st.stop()

model_name = bundle.get('model_name', 'Unknown')
model = bundle['model']
preprocessor = bundle['preprocessor']
FEATURE_COLS = bundle['feature_cols']
area_col = bundle.get('area_col', None)
business_col = bundle.get('business_col', None)

st.write(f"Using model: **{model_name}**")

# -------------------------
# Load CSV dataset
# -------------------------
if CSV_PATH.exists():
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        st.warning(f"Failed to load CSV file: {e}")
        df = pd.DataFrame(columns=FEATURE_COLS)
else:
    st.warning(f"CSV file not found at `{CSV_PATH}`. Proceeding with empty dataset.")
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
    multiselect_areas = st.sidebar.multiselect(
        "Compare areas (pick multiple)", options=all_areas, default=all_areas[:5]
    )
else:
    st.sidebar.info("No 'area' column detected; app will operate per-row.")

top_k = st.sidebar.slider("How many top areas to show (overall)", 1, 20, 5)

# -------------------------
# Prepare features for prediction
# -------------------------
def make_feature_rows_for_areas(df, areas):
    rows = []
    for a in areas:
        subset = df[df[area_col].astype(str) == str(a)] if (area_col and area_col in df.columns) else pd.DataFrame()
        if subset.shape[0] == 0:
            row = {c: np.nan for c in FEATURE_COLS}
            if area_col:
                row[area_col] = a
            if business_col:
                row[business_col] = preferred_business or np.nan
        else:
            row = {}
            for c in FEATURE_COLS:
                if c in subset.columns:
                    if pd.api.types.is_numeric_dtype(subset[c]):
                        row[c] = subset[c].mean()
                    else:
                        row[c] = subset[c].mode().iloc[0] if not subset[c].mode().empty else subset[c].iloc[0]
                else:
                    row[c] = np.nan
            if business_col:
                row[business_col] = preferred_business or row.get(business_col, np.nan)
        # tag preferred business
        if business_col and preferred_business:
            row[business_col] = preferred_business
        rows.append(row)
    return pd.DataFrame(rows)

# Build prediction dataframe
prediction_df = None
if multiselect_areas:
    prediction_df = make_feature_rows_for_areas(df, multiselect_areas)
    labels = multiselect_areas
elif preferred_area and area_col and area_col in df.columns:
    prediction_df = make_feature_rows_for_areas(df, [preferred_area])
    labels = [preferred_area]
else:
    # fallback: top_k aggregated by area
    if area_col and area_col in df.columns:
        area_scores_df = df.groupby(area_col).apply(lambda g: g.mean(numeric_only=True)).reset_index()
        top_areas = area_scores_df.sort_values(by=area_scores_df.columns[1], ascending=False)[area_col].astype(str).tolist()[:top_k]
        prediction_df = make_feature_rows_for_areas(df, top_areas)
        labels = top_areas
    else:
        # operate per-row
        sample = df.head(top_k)
        if sample.shape[0] == 0:
            st.warning("Dataset appears empty or has no usable features.")
            prediction_df = pd.DataFrame([{c: np.nan for c in FEATURE_COLS}])
            labels = ["Unknown"]
        else:
            rows = []
            lbls = []
            for idx, row in sample.iterrows():
                r = {c: row[c] if c in row else np.nan for c in FEATURE_COLS}
                if business_col and preferred_business:
                    r[business_col] = preferred_business
                rows.append(r)
                lbls.append(str(row[area_col]) if area_col in row else f"row-{idx}")
            prediction_df = pd.DataFrame(rows)
            labels = lbls

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
# Display results
# -------------------------
st.subheader("Predicted survival scores")
st.write("Higher score (0-100) → better chance of business surviving there.")
st.dataframe(result_df.reset_index(drop=True))

# Bar chart
st.subheader("Comparison chart")
fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(result_df))))
ax.barh(result_df['area'].astype(str)[::-1], result_df['predicted_survival_score'][::-1])
ax.set_xlabel("Predicted survival score (0-100)")
ax.set_title("Area comparison — higher is better")
plt.tight_layout()
st.pyplot(fig)

# Pros & Cons heuristic
st.subheader("Pros & Cons (automatically inferred from dataset medians)")
proscons = []
medians = {c: df[c].median() for c in FEATURE_COLS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])}

for i, row in result_df.iterrows():
    area_label = row['area']
    idx = list(labels).index(area_label) if area_label in labels else None
    feat_row = prediction_df.iloc[idx] if idx is not None else pd.Series({c: np.nan for c in FEATURE_COLS})

    pros, cons = [], []
    for c in FEATURE_COLS:
        if c not in feat_row.index:
            continue
        try:
            val = float(feat_row[c])
        except Exception:
            continue
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

# Top suggestions
st.subheader("Top suggestions")
st.write(f"Top {min(len(result_df), top_k)} areas by predicted survival score:")
st.table(result_df.head(top_k).reset_index(drop=True))

# Method notes
st.markdown("""
**Method notes:**  
- The underlying model was trained on your dataset (`buzz-complete.csv`) using both Linear Regression and Random Forest.  
- Aggregates area-level features (mean numeric, mode categorical) and predicts a survival score (0–100).  
- This is a data-driven recommendation — quality depends on features in your CSV (population, demand, footfall, rent, crime, etc.).
""")
